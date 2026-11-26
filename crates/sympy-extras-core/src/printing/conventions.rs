//! Port of `sympy_extras/printing/conventions.py`: the conventions of
//! names which the printers share (the scripts of a name, the Greek
//! letters, the modifiers at the end of a name).

// PORTED FROM SYMPY: sympy/printing/conventions.py:split_super_sub (SymPy 1.14, BSD license)

/// The names of the Greek letters (`sympy_extras.core.alphabets.greeks`).
pub const GREEK_NAMES: [&str; 24] = [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu", "nu", "xi",
    "omicron", "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega",
];

/// Whether a (lower-case) name is the name of a Greek letter.
pub fn is_greek(name: &str) -> bool {
    GREEK_NAMES.contains(&name)
}

/// A byte offset at or after `from` where `needle` starts, or `len`.
fn find_from(text: &str, needle: char, from: usize) -> usize {
    if from >= text.len() {
        return text.len();
    }
    match text[from..].find(needle) {
        Some(i) => from + i,
        None => text.len(),
    }
}

/// The name, the superscripts (after `^` or `__`) and the subscripts
/// (after `_`, or the digits at the end of the name) (`split_super_sub`).
pub fn split_super_sub(text: &str) -> (String, Vec<String>, Vec<String>) {
    if text.is_empty() {
        return (String::new(), Vec::new(), Vec::new());
    }
    let mut pos: usize = 0;
    let mut name: String = String::new();
    let mut first: bool = true;
    let mut supers: Vec<String> = Vec::new();
    let mut subs: Vec<String> = Vec::new();
    while pos < text.len() {
        // one character on (a byte offset: the separators are ASCII)
        let mut start: usize = pos + text[pos..].chars().next().map_or(1, char::len_utf8);
        if text[pos..].starts_with("__") {
            start += 1;
        }
        let pos_next: usize = find_from(text, '^', start).min(find_from(text, '_', start));
        let part: &str = &text[pos..pos_next];
        pos = pos_next;
        if first {
            name = part.to_string();
            first = false;
        } else if let Some(rest) = part.strip_prefix('^') {
            supers.push(rest.to_string());
        } else if let Some(rest) = part.strip_prefix("__") {
            supers.push(rest.to_string());
        } else {
            subs.push(part.chars().skip(1).collect());
        }
    }
    // a name of letters followed by digits: the digits are a subscript
    let digits: String = name.chars().rev().take_while(|c| c.is_ascii_digit()).collect();
    if !digits.is_empty() {
        let letters: String = name
            .chars()
            .take(name.chars().count() - digits.chars().count())
            .collect();
        if !letters.is_empty() && letters.chars().all(|c| c.is_alphabetic()) {
            let digits: String = digits.chars().rev().collect();
            subs.insert(0, digits);
            name = letters;
        }
    }
    (name, supers, subs)
}

/// A modifier at the end of a name and its form.
pub type Modifier = (&'static str, fn(&str) -> String);

/// A name without scripts in the notation of a printer: `letter` gives the
/// form of a whole name which has one (a Greek letter), `modifiers` the
/// forms of the modifiers at the end of a name (`xdot`, `xhat`), tried in
/// their order and applied to the translation of what precedes them; any
/// other name is itself (`translate_name`).
pub fn translate_name(name: &str, letter: &dyn Fn(&str) -> Option<String>, modifiers: &[Modifier]) -> String {
    if let Some(whole) = letter(name) {
        return whole;
    }
    let lowered: String = name.to_lowercase();
    for (key, modifier) in modifiers {
        if lowered.ends_with(key) && name.chars().count() > key.chars().count() {
            let head: String = name.chars().take(name.chars().count() - key.chars().count()).collect();
            return modifier(&translate_name(&head, letter, modifiers));
        }
    }
    name.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scripts() {
        let s = |t: &str| {
            let (n, a, b) = split_super_sub(t);
            (n, a, b)
        };
        assert_eq!(
            s("a_x^1"),
            ("a".to_string(), vec!["1".to_string()], vec!["x".to_string()])
        );
        assert_eq!(
            s("var_sub1__sup_sub2"),
            (
                "var".to_string(),
                vec!["sup".to_string()],
                vec!["sub1".to_string(), "sub2".to_string()]
            )
        );
        assert_eq!(s("x12"), ("x".to_string(), vec![], vec!["12".to_string()]));
    }
}
