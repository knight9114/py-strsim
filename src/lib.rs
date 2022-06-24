use pyo3::prelude::*;
use pyo3::exceptions::PyOSError;
use rayon::prelude::*;

// ------------------------------------------------------------------------
//  Direct `strsim` Bindings
// ------------------------------------------------------------------------

pub mod single {
    use super::*;

    /// Like optimal string alignment, but substrings can be edited an unlimited
    /// number of times, and the triangle inequality holds.
    /// 
    /// # Arguments
    /// 
    /// * `a` - First string to compare
    /// * `b` - Secondary string to compare to `a`
    /// 
    /// # Returns
    /// 
    /// * `output` - Distance between `a` and `b`
    #[pyfunction]
    #[pyo3(text_signature = "(a, b, /)")]
    pub fn damerau_levenshtein(a: &str, b: &str) -> usize {
        strsim::damerau_levenshtein(a, b)
    }

    /// Calculates the Jaro similarity between two strings. The returned value
    /// is between 0.0 and 1.0 (higher value means more similar).
    /// 
    /// # Arguments
    /// 
    /// * `a` - First string to compare
    /// * `b` - Secondary string to compare to `a`
    /// 
    /// # Returns
    /// 
    /// * `output` - Similarity between `a` and `b`
    #[pyfunction]
    #[pyo3(text_signature = "(a, b, /)")]
    pub fn jaro(a: &str, b: &str) -> f64 {
        strsim::jaro(a, b)
    }

    /// Like Jaro but gives a boost to strings that have a common prefix.
    /// 
    /// # Arguments
    /// 
    /// * `a` - First string to compare
    /// * `b` - Secondary string to compare to `a`
    /// 
    /// # Returns
    /// 
    /// * `output` - Similarity between `a` and `b`
    #[pyfunction]
    #[pyo3(text_signature = "(a, b, /)")]
    pub fn jaro_winkler(a: &str, b: &str) -> f64 {
        strsim::jaro_winkler(a, b)
    }

    /// Calculates the minimum number of insertions, deletions, and substitutions
    /// required to change one string into the other.
    /// 
    /// # Arguments
    /// 
    /// * `a` - First string to compare
    /// * `b` - Secondary string to compare to `a`
    /// 
    /// # Returns
    /// 
    /// * `output` - Distance between `a` and `b`
    #[pyfunction]
    #[pyo3(text_signature = "(a, b, /)")]
    pub fn levenshtein(a: &str, b: &str) -> usize {
        strsim::levenshtein(a, b)
    }

    /// Calculates a normalized score of the Damerau–Levenshtein algorithm between
    /// 0.0 and 1.0 (inclusive), where 1.0 means the strings are the same.
    /// 
    /// # Arguments
    /// 
    /// * `a` - First string to compare
    /// * `b` - Secondary string to compare to `a`
    /// 
    /// # Returns
    /// 
    /// * `output` - Similarity between `a` and `b`
    #[pyfunction]
    #[pyo3(text_signature = "(a, b, /)")]
    pub fn normalized_damerau_levenshtein(a: &str, b: &str) -> f64 {
        strsim::normalized_damerau_levenshtein(a, b)
    }

    /// Calculates a normalized score of the Levenshtein algorithm between 0.0 and
    /// 1.0 (inclusive), where 1.0 means the strings are the same.
    /// 
    /// # Arguments
    /// 
    /// * `a` - First string to compare
    /// * `b` - Secondary string to compare to `a`
    /// 
    /// # Returns
    /// 
    /// * `output` - Similarity between `a` and `b`
    #[pyfunction]
    #[pyo3(text_signature = "(a, b, /)")]
    pub fn normalized_levenshtein(a: &str, b: &str) -> f64 {
        strsim::normalized_levenshtein(a, b)
    }

    /// Like Levenshtein but allows for adjacent transpositions. Each substring can
    /// only be edited once.
    /// 
    /// # Arguments
    /// 
    /// * `a` - First string to compare
    /// * `b` - Secondary string to compare to `a`
    /// 
    /// # Returns
    /// 
    /// * `output` - Distance between `a` and `b`
    #[pyfunction]
    #[pyo3(text_signature = "(a, b, /)")]
    pub fn osa_distance(a: &str, b: &str) -> usize {
        strsim::osa_distance(a, b)
    }

    /// Calculates a Sørensen-Dice similarity distance using bigrams.
    /// See http://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient.
    /// 
    /// # Arguments
    /// 
    /// * `a` - First string to compare
    /// * `b` - Secondary string to compare to `a`
    /// 
    /// # Returns
    /// 
    /// * `output` - Similarity between `a` and `b`
    #[pyfunction]
    #[pyo3(text_signature = "(a, b, /)")]
    pub fn sorensen_dice(a: &str, b: &str) -> f64 {
        strsim::sorensen_dice(a, b)
    }
}


// ------------------------------------------------------------------------
//  Composite Functions
// ------------------------------------------------------------------------

pub mod vectorized {
    use super::*;

    fn create_thread_pool(n: usize) -> PyResult<rayon::ThreadPool> {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .map_err(|_| PyOSError::new_err("failed to allocate threads"))
    }

    fn vectorize<F: Send + Sync>(f: fn(&str, &str) -> F, n: usize, a: &str, bs: Vec<&str>) -> PyResult<Vec<F>> {
        Ok(
            create_thread_pool(n)?
                .install(|| {
                    bs
                        .par_iter()
                        .map(|&b| f(a, b))
                        .collect()
                })
        )
    }

    /// Like optimal string alignment, but substrings can be edited an unlimited
    /// number of times, and the triangle inequality holds.
    /// 
    /// # Arguments
    /// 
    /// * `n` - Number of threads to use
    /// * `a` - First string to compare
    /// * `bs` - Secondary strings to compare to `a`
    /// 
    /// # Returns
    /// 
    /// * `output` - Distances between `a` and each `b` in `bs`
    #[pyfunction]
    #[pyo3(text_signature = "(n, a, bs, /)")]
    pub fn damerau_levenshtein(n: usize, a: &str, bs: Vec<&str>) -> PyResult<Vec<usize>> {
        vectorize::<usize>(strsim::damerau_levenshtein, n, a, bs)
    }

    /// Calculates the Jaro similarity between two strings. The returned value
    /// is between 0.0 and 1.0 (higher value means more similar).
    /// 
    /// # Arguments
    /// 
    /// * `n` - Number of threads to use
    /// * `a` - First string to compare
    /// * `bs` - Secondary strings to compare to `a`
    /// 
    /// # Returns
    /// 
    /// * `output` - Similarities between `a` and each `b` in `bs`
    #[pyfunction]
    #[pyo3(text_signature = "(n, a, bs, /)")]
    pub fn jaro(n: usize, a: &str, bs: Vec<&str>) -> PyResult<Vec<f64>> {
        vectorize::<f64>(strsim::jaro, n, a, bs)
    }

    /// Like Jaro but gives a boost to strings that have a common prefix.
    /// 
    /// # Arguments
    /// 
    /// * `n` - Number of threads to use
    /// * `a` - First string to compare
    /// * `bs` - Secondary strings to compare to `a`
    /// 
    /// # Returns
    /// 
    /// * `output` - Similarities between `a` and each `b` in `bs`
    #[pyfunction]
    #[pyo3(text_signature = "(n, a, bs, /)")]
    pub fn jaro_winkler(n: usize, a: &str, bs: Vec<&str>) -> PyResult<Vec<f64>> {
        vectorize::<f64>(strsim::jaro_winkler, n, a, bs)
    }

    /// Calculates the minimum number of insertions, deletions, and substitutions
    /// required to change one string into the other.
    /// 
    /// # Arguments
    /// 
    /// * `n` - Number of threads to use
    /// * `a` - First string to compare
    /// * `bs` - Secondary strings to compare to `a`
    /// 
    /// # Returns
    /// 
    /// * `output` - Distances between `a` and each `b` in `bs`
    #[pyfunction]
    #[pyo3(text_signature = "(n, a, bs, /)")]
    pub fn levenshtein(n: usize, a: &str, bs: Vec<&str>) -> PyResult<Vec<usize>> {
        vectorize::<usize>(strsim::levenshtein, n, a, bs)
    }

    /// Calculates a normalized score of the Damerau–Levenshtein algorithm between
    /// 0.0 and 1.0 (inclusive), where 1.0 means the strings are the same.
    /// 
    /// # Arguments
    /// 
    /// * `n` - Number of threads to use
    /// * `a` - First string to compare
    /// * `bs` - Secondary strings to compare to `a`
    /// 
    /// # Returns
    /// 
    /// * `output` - Similarities between `a` and each `b` in `bs`
    #[pyfunction]
    #[pyo3(text_signature = "(n, a, bs, /)")]
    pub fn normalized_damerau_levenshtein(n: usize, a: &str, bs: Vec<&str>) -> PyResult<Vec<f64>> {
        vectorize::<f64>(strsim::normalized_damerau_levenshtein, n, a, bs)
    }

    /// Calculates a normalized score of the Levenshtein algorithm between 0.0 and
    /// 1.0 (inclusive), where 1.0 means the strings are the same.
    /// 
    /// # Arguments
    /// 
    /// * `n` - Number of threads to use
    /// * `a` - First string to compare
    /// * `bs` - Secondary strings to compare to `a`
    /// 
    /// # Returns
    /// 
    /// * `output` - Similarities between `a` and each `b` in `bs`
    #[pyfunction]
    #[pyo3(text_signature = "(n, a, bs, /)")]
    pub fn normalized_levenshtein(n: usize, a: &str, bs: Vec<&str>) -> PyResult<Vec<f64>> {
        vectorize::<f64>(strsim::normalized_levenshtein, n, a, bs)
    }

    /// Like Levenshtein but allows for adjacent transpositions. Each substring can
    /// only be edited once.
    /// 
    /// # Arguments
    /// 
    /// * `n` - Number of threads to use
    /// * `a` - First string to compare
    /// * `bs` - Secondary strings to compare to `a`
    /// 
    /// # Returns
    /// 
    /// * `output` - Distances between `a` and each `b` in `bs`
    #[pyfunction]
    #[pyo3(text_signature = "(n, a, bs, /)")]
    pub fn osa_distance(n: usize, a: &str, bs: Vec<&str>) -> PyResult<Vec<usize>> {
        vectorize::<usize>(strsim::osa_distance, n, a, bs)
    }

    /// Calculates a Sørensen-Dice similarity distance using bigrams.
    /// See http://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient.
    /// 
    /// # Arguments
    /// 
    /// * `n` - Number of threads to use
    /// * `a` - First string to compare
    /// * `bs` - Secondary strings to compare to `a`
    /// 
    /// # Returns
    /// 
    /// * `output` - Similarities between `a` and each `b` in `bs`
    #[pyfunction]
    #[pyo3(text_signature = "(n, a, bs, /)")]
    pub fn sorensen_dice(n: usize, a: &str, bs: Vec<&str>) -> PyResult<Vec<f64>> {
        vectorize::<f64>(strsim::sorensen_dice, n, a, bs)
    }
}


// ------------------------------------------------------------------------
//  Module Declarations
// ------------------------------------------------------------------------

#[pymodule]
#[pyo3(name = "_py_strsim")]
fn py_strsim(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    register_child_modules(py, m)?;
    Ok(())
}

fn register_child_modules(py: Python<'_>, parent: &PyModule) -> PyResult<()> {
    let single_module = PyModule::new(py, "single")?;
    single_module.add_function(wrap_pyfunction!(single::damerau_levenshtein, single_module)?)?;
    single_module.add_function(wrap_pyfunction!(single::jaro, single_module)?)?;
    single_module.add_function(wrap_pyfunction!(single::jaro_winkler, single_module)?)?;
    single_module.add_function(wrap_pyfunction!(single::levenshtein, single_module)?)?;
    single_module.add_function(wrap_pyfunction!(single::normalized_levenshtein, single_module)?)?;
    single_module.add_function(wrap_pyfunction!(single::normalized_damerau_levenshtein, single_module)?)?;
    single_module.add_function(wrap_pyfunction!(single::osa_distance, single_module)?)?;
    single_module.add_function(wrap_pyfunction!(single::sorensen_dice, single_module)?)?;

    let vectorized_module = PyModule::new(py, "vectorized")?;
    vectorized_module.add_function(wrap_pyfunction!(vectorized::damerau_levenshtein, vectorized_module)?)?;
    vectorized_module.add_function(wrap_pyfunction!(vectorized::jaro, vectorized_module)?)?;
    vectorized_module.add_function(wrap_pyfunction!(vectorized::jaro_winkler, vectorized_module)?)?;
    vectorized_module.add_function(wrap_pyfunction!(vectorized::levenshtein, vectorized_module)?)?;
    vectorized_module.add_function(wrap_pyfunction!(vectorized::normalized_levenshtein, vectorized_module)?)?;
    vectorized_module.add_function(wrap_pyfunction!(vectorized::normalized_damerau_levenshtein, vectorized_module)?)?;
    vectorized_module.add_function(wrap_pyfunction!(vectorized::osa_distance, vectorized_module)?)?;
    vectorized_module.add_function(wrap_pyfunction!(vectorized::sorensen_dice, vectorized_module)?)?;

    parent.add_submodule(single_module)?;
    parent.add_submodule(vectorized_module)?;

    Ok(())
}