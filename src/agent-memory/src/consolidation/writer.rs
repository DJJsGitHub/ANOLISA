//! Fact writer — writes consolidated facts to the filesystem.
//!
//! Produces two outputs per fact:
//! 1. `facts/<category>/<ulid>.md` — markdown with YAML frontmatter
//! 2. `facts/facts.jsonl` — JSONL line appended to the same directory
//!
//! All writes use `safe_fs` (openat2 + RESOLVE_BENEATH) when a root_fd
//! is provided, ensuring namespace sandbox containment.

use std::fs::OpenOptions;
use std::io::Write;
use std::os::fd::{AsFd, OwnedFd};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::error::Result;
use crate::index::store::BM25Store;

use super::fact::ConsolidatedFact;

/// Writes facts to a given base directory.
pub struct FactWriter {
    facts_dir: PathBuf,
    jsonl_path: PathBuf,
    /// Mount root fd for sandboxed writes via safe_fs (openat2 + RESOLVE_BENEATH).
    /// None falls back to std::fs (used in tests with temp dirs).
    root_fd: Option<Arc<OwnedFd>>,
    /// Held file handle for fallback (non-sandboxed) JSONL writes.
    jsonl_file: Mutex<Option<std::fs::File>>,
    /// Optional BM25 store for conflict detection.
    index: Option<Arc<Mutex<BM25Store>>>,
    /// BM25 threshold for conflict detection.
    conflict_threshold: f64,
}

impl FactWriter {
    pub fn new(base_dir: &Path) -> Self {
        let facts_dir = base_dir.join("facts");
        let jsonl_path = facts_dir.join("facts.jsonl");
        Self {
            facts_dir,
            jsonl_path,
            root_fd: None,
            jsonl_file: Mutex::new(None),
            index: None,
            conflict_threshold: -2.0,
        }
    }

    /// Attach a root fd for sandboxed writes.
    pub fn with_root_fd(mut self, fd: Arc<OwnedFd>) -> Self {
        self.root_fd = Some(fd);
        self
    }

    pub fn with_index(mut self, index: Arc<Mutex<BM25Store>>, conflict_threshold: f64) -> Self {
        self.index = Some(index);
        self.conflict_threshold = conflict_threshold;
        self
    }

    /// Ensure the facts directory exists.
    pub fn ensure_dir(&self) -> Result<()> {
        std::fs::create_dir_all(&self.facts_dir)?;
        Ok(())
    }

    /// Write a single fact: creates `<category>/<ulid>.md` and appends to `facts.jsonl`.
    /// Uses safe_fs (openat2 + RESOLVE_BENEATH) when root_fd is available,
    /// falling back to std::fs for tests with temp dirs.
    /// If conflict detection is enabled, marks similar existing facts as superseded.
    /// Facts are organized into category subdirectories under facts/.
    ///
    /// The conflicting facts are retired only once the replacement is durable
    /// (markdown on disk + JSONL appended). `supersede()` runs an autocommitted
    /// `UPDATE files SET is_superseded = 1` and every search path filters on
    /// `is_superseded = 0`, so superseding first would drop the old fact out of
    /// recall whenever a later step fails and no replacement ever lands (#4406).
    pub fn write(&self, fact: &ConsolidatedFact) -> Result<()> {
        std::fs::create_dir_all(&self.facts_dir)?;

        // Conflict detection: search for similar facts before writing, but only
        // record what they are. The scan is read-only and has to stay ahead of
        // the write: once the new markdown is on disk the index worker can pick
        // it up, and a later scan could match the fact against itself.
        let conflicts = self.find_conflicts(fact);

        // Write markdown file under category subdirectory.
        let category_dir = self.facts_dir.join(fact.category.to_string());
        std::fs::create_dir_all(&category_dir)?;
        let md_path = category_dir.join(format!("{}.md", fact.id));

        if let Some(ref fd) = self.root_fd {
            // Sandboxed write via safe_fs (openat2 + RESOLVE_BENEATH).
            let md_rel = Path::new("facts")
                .join(fact.category.to_string())
                .join(format!("{}.md", fact.id));
            crate::safe_fs::write(fd.as_fd(), &md_rel, fact.to_markdown().as_bytes())?;
        } else {
            std::fs::write(&md_path, fact.to_markdown())?;
        }

        // Append JSONL line.
        let jsonl_str = match fact.to_jsonl() {
            Ok(s) => s,
            Err(e) => {
                // The markdown landed but the JSONL feed will not, so the
                // replacement is not durable: leave the conflicting facts
                // searchable instead of retiring them (#4406).
                tracing::warn!("failed to serialize fact {} to JSONL: {e}", fact.id);
                return Ok(());
            }
        };
        let jsonl_line = format!("{jsonl_str}\n");
        if let Some(ref fd) = self.root_fd {
            let jsonl_rel = Path::new("facts").join("facts.jsonl");
            crate::safe_fs::append(fd.as_fd(), &jsonl_rel, jsonl_line.as_bytes())?;
        } else {
            let mut guard = self.jsonl_file.lock().unwrap_or_else(|e| e.into_inner());
            if guard.is_none() {
                let f = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&self.jsonl_path)?;
                *guard = Some(f);
            }
            let f = guard.as_mut().expect("just inserted");
            f.write_all(jsonl_line.as_bytes())?;
            f.sync_all()?;
        }

        // Both outputs are on disk now, so retiring the facts this one
        // replaces can no longer orphan the knowledge they held.
        self.supersede_conflicts(fact, &conflicts);

        tracing::debug!("wrote fact: {}", md_path.display());
        Ok(())
    }

    /// Read-only conflict scan: the existing facts that `fact` should replace.
    /// Returns nothing when conflict detection is disabled, and a failed scan
    /// degrades to "no conflicts" rather than blocking the write.
    fn find_conflicts(&self, fact: &ConsolidatedFact) -> Vec<(String, f64)> {
        let Some(store) = self.index.as_ref() else {
            return Vec::new();
        };
        let search_text = format!(
            "{} {}",
            fact.title,
            fact.content.chars().take(100).collect::<String>()
        );
        let s = store.lock().unwrap_or_else(|e| e.into_inner());
        match s.detect_conflicts(&search_text, self.conflict_threshold) {
            Ok(conflicts) => conflicts,
            Err(e) => {
                tracing::warn!("conflict detection failed: {e}");
                Vec::new()
            }
        }
    }

    /// Mark the facts found by [`Self::find_conflicts`] as superseded. Called
    /// only after the replacement has been persisted.
    fn supersede_conflicts(&self, fact: &ConsolidatedFact, conflicts: &[(String, f64)]) {
        if conflicts.is_empty() {
            return;
        }
        let Some(store) = self.index.as_ref() else {
            return;
        };
        let mut s = store.lock().unwrap_or_else(|e| e.into_inner());
        for (old_path, score) in conflicts {
            tracing::info!(
                "conflict detected: new fact '{}' conflicts with '{}' (score={:.2})",
                fact.title,
                old_path,
                score
            );
            let _ = s.supersede(old_path, &fact.id);
        }
    }

    /// Write multiple facts in one batch.
    pub fn write_batch(&self, facts: &[ConsolidatedFact]) -> Result<usize> {
        if facts.is_empty() {
            return Ok(0);
        }
        self.ensure_dir()?;

        let mut written = 0;
        for fact in facts {
            if let Err(e) = self.write(fact) {
                tracing::warn!("failed to write fact {}: {e}", fact.id);
            } else {
                written += 1;
            }
        }

        tracing::info!("batch write: {written}/{} facts written", facts.len());
        Ok(written)
    }

    pub fn facts_dir(&self) -> &Path {
        &self.facts_dir
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consolidation::fact::{ConsolidatedFact, FactCategory};

    #[test]
    fn write_single_fact() {
        let tmp = tempfile::tempdir().unwrap();
        let writer = FactWriter::new(tmp.path());
        let fact = ConsolidatedFact::new(
            "test-sid",
            FactCategory::WorkingContext,
            "Test fact".into(),
            "Test content body".into(),
            "mem_write".into(),
            vec!["notes/a.md".into()],
            0.8,
        );
        writer.write(&fact).unwrap();

        let md_path = tmp
            .path()
            .join("facts")
            .join("working-context")
            .join(format!("{}.md", fact.id));
        assert!(md_path.exists());
        assert!(
            std::fs::read_to_string(&md_path)
                .unwrap()
                .contains("Test content")
        );

        let jsonl_path = tmp.path().join("facts").join("facts.jsonl");
        assert!(jsonl_path.exists());
        let content = std::fs::read_to_string(&jsonl_path).unwrap();
        let lines: Vec<_> = content.lines().collect();
        assert_eq!(lines.len(), 1);
        let parsed: ConsolidatedFact = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(parsed.id, fact.id);
    }

    #[test]
    fn write_batch_dedup() {
        let tmp = tempfile::tempdir().unwrap();
        let writer = FactWriter::new(tmp.path());
        let facts = vec![
            ConsolidatedFact::new(
                "s1",
                FactCategory::Interest,
                "A".into(),
                "Content A".into(),
                "mem_search".into(),
                vec![],
                0.5,
            ),
            ConsolidatedFact::new(
                "s2",
                FactCategory::Lesson,
                "B".into(),
                "Content B".into(),
                "mem_edit".into(),
                vec![],
                0.6,
            ),
        ];
        let n = writer.write_batch(&facts).unwrap();
        assert_eq!(n, 2);
        let jsonl_path = tmp.path().join("facts").join("facts.jsonl");
        let content = std::fs::read_to_string(&jsonl_path).unwrap();
        let lines: Vec<_> = content.lines().collect();
        assert_eq!(lines.len(), 2);
    }

    /// A store holding one fact that a new `Lesson` fact conflicts with, plus a
    /// writer whose conflict threshold sits far below any BM25 score — these
    /// tests pin the supersede *ordering*, not the detection tuning.
    fn conflicting_store_and_writer(
        tmp: &std::path::Path,
    ) -> (Arc<Mutex<BM25Store>>, FactWriter, ConsolidatedFact) {
        let store = Arc::new(Mutex::new(BM25Store::open_in_memory().unwrap()));
        store
            .lock()
            .unwrap()
            .upsert(
                "notes/old-lesson.md",
                100,
                50,
                "borrow checker rejects nested closure capture",
                None,
            )
            .unwrap();

        let writer = FactWriter::new(tmp).with_index(Arc::clone(&store), -1_000_000.0);
        let fact = ConsolidatedFact::new(
            "test-sid",
            FactCategory::Lesson,
            "borrow checker rejects nested closure capture".into(),
            "borrow checker rejects nested closure capture".into(),
            "mem_edit".into(),
            vec!["src/lib.rs".into()],
            0.9,
        );
        (store, writer, fact)
    }

    /// #4406 regression: `supersede()` is an autocommitted
    /// `UPDATE files SET is_superseded = 1` and every search path filters on
    /// `is_superseded = 0`. Running it before the replacement was durable meant
    /// a failed write retired the conflicting fact without landing the new one,
    /// so the knowledge vanished from recall with nothing in its place.
    #[test]
    fn write_failure_keeps_conflicting_fact_searchable() {
        let tmp = tempfile::tempdir().unwrap();
        let (store, writer, fact) = conflicting_store_and_writer(tmp.path());

        // Block `facts/lesson/` with a regular file so the markdown write fails.
        let facts_dir = tmp.path().join("facts");
        std::fs::create_dir_all(&facts_dir).unwrap();
        std::fs::write(facts_dir.join("lesson"), "not a directory").unwrap();

        // Sanity: the scan does find the old fact, so the assertion below can
        // only fail because of when supersede runs.
        let conflicts = writer.find_conflicts(&fact);
        assert_eq!(
            conflicts.iter().map(|(p, _)| p).collect::<Vec<_>>(),
            vec!["notes/old-lesson.md"]
        );

        let res = writer.write(&fact);
        assert!(res.is_err(), "category dir is blocked, write must fail");

        let hits = store
            .lock()
            .unwrap()
            .search("borrow checker", 10, true)
            .unwrap();
        assert!(
            hits.iter().any(|h| h.path == "notes/old-lesson.md"),
            "old fact lost from recall after a failed write: {:?}",
            hits.iter().map(|h| h.path.clone()).collect::<Vec<_>>()
        );
    }

    /// The other half of #4406: once the replacement *is* durable the
    /// conflicting fact must still be retired, or both would answer searches.
    #[test]
    fn successful_write_supersedes_conflicting_fact() {
        let tmp = tempfile::tempdir().unwrap();
        let (store, writer, fact) = conflicting_store_and_writer(tmp.path());

        writer.write(&fact).unwrap();

        let md_path = tmp
            .path()
            .join("facts")
            .join("lesson")
            .join(format!("{}.md", fact.id));
        assert!(md_path.exists(), "replacement markdown must be on disk");

        let hits = store
            .lock()
            .unwrap()
            .search("borrow checker", 10, true)
            .unwrap();
        assert!(
            hits.iter().all(|h| h.path != "notes/old-lesson.md"),
            "old fact still searchable after a successful write: {:?}",
            hits.iter().map(|h| h.path.clone()).collect::<Vec<_>>()
        );
    }
}
