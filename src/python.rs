use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::sync::Arc;
use tokio::runtime::Runtime;

use crate::{local, IndexReader};

#[pyclass]
pub struct PySearchResult {
    #[pyo3(get)]
    pub doc_id: u32,
    #[pyo3(get)]
    pub score: f32,
    #[pyo3(get)]
    pub text: String,
    #[pyo3(get)]
    pub attributes: std::collections::HashMap<String, String>,
}

impl From<crate::types::SearchResult> for PySearchResult {
    fn from(result: crate::types::SearchResult) -> Self {
        Self {
            doc_id: result.doc_id,
            score: result.score,
            text: result.text,
            attributes: result.attributes,
        }
    }
}

#[pyclass]
pub struct PyIndex {
    rt: Runtime,
    reader: IndexReader,
}

#[pymethods]
impl PyIndex {
    #[new]
    pub fn new(index_dir: String, version: String) -> PyResult<Self> {
        let rt = Runtime::new().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let store = Arc::new(local(index_dir).map_err(|e| PyRuntimeError::new_err(e.to_string()))?);
        let reader = rt
            .block_on(IndexReader::open(store, &version))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(Self { rt, reader })
    }

    #[getter]
    pub fn doc_count(&self) -> u32 {
        self.reader.doc_count()
    }

    pub fn search_lexical(&self, query: String, top_k: usize) -> PyResult<Vec<PySearchResult>> {
        let results = self
            .rt
            .block_on(self.reader.search_lexical(&query, top_k))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(results.into_iter().map(PySearchResult::from).collect())
    }

    pub fn search_vector(
        &self,
        embedding: Vec<f32>,
        top_k: usize,
        probes: usize,
    ) -> PyResult<Vec<PySearchResult>> {
        let results = self
            .rt
            .block_on(self.reader.search_vector(&embedding, top_k, probes))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(results.into_iter().map(PySearchResult::from).collect())
    }

    pub fn search_hybrid(
        &self,
        query: String,
        embedding: Vec<f32>,
        top_k: usize,
        probes: usize,
    ) -> PyResult<Vec<PySearchResult>> {
        let results = self
            .rt
            .block_on(self.reader.search_hybrid(&query, &embedding, top_k, probes))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(results.into_iter().map(PySearchResult::from).collect())
    }
}

#[pymodule]
pub fn seglens(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyIndex>()?;
    m.add_class::<PySearchResult>()?;
    Ok(())
}
