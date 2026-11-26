//! Port of `sympy_extras/printing/_latex_names.py`: the LaTeX of names
//! (symbols with their subscripts, superscripts, Greek letters and
//! modifiers, the names of functions, text).

// PORTED FROM SYMPY: sympy/printing/latex.py:translate, latex_escape, LatexPrinter._deal_with_super_sub
// (SymPy 1.14, BSD license; the special characters of TeX escaped in the names, sympy/sympy#30844)

use super::conventions::{is_greek, split_super_sub, translate_name, Modifier};

/// The functions which are commands of LaTeX and MathJax.
pub const ACCEPTED_FUNCTIONS: [&str; 16] = [
    "arcsin", "arccos", "arctan", "sin", "cos", "tan", "sinh", "cosh", "tanh", "ln", "log", "sec", "csc", "cot",
    "coth", "arg",
];

fn greek(s: &str) -> Option<&'static str> {
    Some(match s {
        "Alpha" => r"\mathrm{A}",
        "Beta" => r"\mathrm{B}",
        "Gamma" => r"\Gamma",
        "Delta" => r"\Delta",
        "Epsilon" => r"\mathrm{E}",
        "Zeta" => r"\mathrm{Z}",
        "Eta" => r"\mathrm{H}",
        "Theta" => r"\Theta",
        "Iota" => r"\mathrm{I}",
        "Kappa" => r"\mathrm{K}",
        "Lambda" => r"\Lambda",
        "Mu" => r"\mathrm{M}",
        "Nu" => r"\mathrm{N}",
        "Xi" => r"\Xi",
        "omicron" => "o",
        "Omicron" => r"\mathrm{O}",
        "Pi" => r"\Pi",
        "Rho" => r"\mathrm{P}",
        "Sigma" => r"\Sigma",
        "Tau" => r"\mathrm{T}",
        "Upsilon" => r"\Upsilon",
        "Phi" => r"\Phi",
        "Chi" => r"\mathrm{X}",
        "Psi" => r"\Psi",
        "Omega" => r"\Omega",
        "lamda" => r"\lambda",
        "Lamda" => r"\Lambda",
        "khi" => r"\chi",
        "Khi" => r"\mathrm{X}",
        "varepsilon" => r"\varepsilon",
        "varkappa" => r"\varkappa",
        "varphi" => r"\varphi",
        "varpi" => r"\varpi",
        "varrho" => r"\varrho",
        "varsigma" => r"\varsigma",
        "vartheta" => r"\vartheta",
        _ => return None,
    })
}

const OTHER_SYMBOLS: [&str; 10] = [
    "aleph", "beth", "daleth", "gimel", "ell", "eth", "hbar", "hslash", "mho", "wp",
];

fn prime(s: &str) -> String {
    format!("{{{}}}'", s)
}

macro_rules! wrap {
    ($name:ident, $command:expr) => {
        fn $name(s: &str) -> String {
            format!("\\{}{{{}}}", $command, s)
        }
    };
}

wrap!(mathring, "mathring");
wrap!(ddddot, "ddddot");
wrap!(dddot, "dddot");
wrap!(ddot, "ddot");
wrap!(dot, "dot");
wrap!(check, "check");
wrap!(breve, "breve");
wrap!(acute, "acute");
wrap!(grave, "grave");
wrap!(tilde, "tilde");
wrap!(hat, "hat");
wrap!(bar, "bar");
wrap!(vec_, "vec");
wrap!(boldsymbol, "boldsymbol");
wrap!(mathcal, "mathcal");
wrap!(mathscr, "mathscr");
wrap!(mathfrak, "mathfrak");

fn norm(s: &str) -> String {
    format!(r"\left\|{{{}}}\right\|", s)
}
fn avg(s: &str) -> String {
    format!(r"\left\langle{{{}}}\right\rangle", s)
}
fn abs_(s: &str) -> String {
    format!(r"\left|{{{}}}\right|", s)
}

/// The modifiers, the longest names first (Python's `sorted(key=len,
/// reverse=True)` of the dict in its insertion order: a stable sort).
fn modifier_table() -> Vec<Modifier> {
    let mut table: Vec<Modifier> = vec![
        ("mathring", mathring),
        ("ddddot", ddddot),
        ("dddot", dddot),
        ("ddot", ddot),
        ("dot", dot),
        ("check", check),
        ("breve", breve),
        ("acute", acute),
        ("grave", grave),
        ("tilde", tilde),
        ("hat", hat),
        ("bar", bar),
        ("vec", vec_),
        ("prime", prime),
        ("prm", prime),
        ("bold", boldsymbol),
        ("bm", boldsymbol),
        ("cal", mathcal),
        ("scr", mathscr),
        ("frak", mathfrak),
        ("norm", norm),
        ("avg", avg),
        ("abs", abs_),
        ("mag", abs_),
    ];
    table.sort_by_key(|(key, _)| std::cmp::Reverse(key.len()));
    table
}

/// The command of a name which is a letter.
fn letter(s: &str) -> Option<String> {
    if let Some(g) = greek(s) {
        return Some(g.to_string());
    }
    let lowered: String = s.to_lowercase();
    if is_greek(&lowered) {
        return Some(format!("\\{}", lowered));
    }
    if OTHER_SYMBOLS.contains(&s) {
        return Some(format!("\\{}", s));
    }
    None
}

/// The LaTeX of a name without scripts: the Greek letters, the modifiers
/// at the end of the name (`xdot`, `xhat`, `xbold`) (`translate`).
pub fn translate(s: &str) -> String {
    translate_name(s, &letter, &modifier_table())
}

fn balanced(s: &str) -> bool {
    let mut depth: i64 = 0;
    for c in s.chars() {
        if c == '{' {
            depth += 1;
        } else if c == '}' {
            depth -= 1;
            if depth < 0 {
                return false;
            }
        }
    }
    depth == 0
}

/// `s` with each `c` not preceded by a backslash replaced by `escaped`.
fn replace_unescaped(s: &str, c: char, escaped: &str) -> String {
    let mut out: String = String::new();
    let mut previous: Option<char> = None;
    for ch in s.chars() {
        if ch == c && previous != Some('\\') {
            out.push_str(escaped);
        } else {
            out.push(ch);
        }
        previous = Some(ch);
    }
    out
}

/// A name with the characters which would break the formula escaped: `%`,
/// `#`, `&`, `$`, `~` always, the braces when they are not balanced; a
/// backslash is kept (`escape_name`).
pub fn escape_name(s: &str) -> String {
    let mut s: String = s.to_string();
    for (c, escaped) in [('%', r"\%"), ('#', r"\#"), ('&', r"\&"), ('$', r"\$"), ('~', r"\sim ")] {
        if s.contains(c) {
            s = replace_unescaped(&s, c, escaped);
        }
    }
    if !balanced(&s) {
        s = replace_unescaped(&s, '{', r"\{");
        s = replace_unescaped(&s, '}', r"\}");
    }
    s
}

/// The LaTeX of the name of a symbol and whether it carries a superscript:
/// the scripts split and translated (`x_1^2`, `alpha_i`); a name with
/// braces is LaTeX already and is kept (`name_latex`).
pub fn name_latex(name: &str) -> (String, bool) {
    if name.contains('{') {
        let escaped: String = escape_name(name);
        let sup: bool = escaped.contains('^');
        return (escaped, sup);
    }
    let (base, supers, subs) = split_super_sub(name);
    let mut tex: String = escape_name(&translate(&base));
    if !supers.is_empty() {
        let parts: Vec<String> = supers.iter().map(|s| escape_name(&translate(s))).collect();
        tex.push_str(&format!("^{{{}}}", parts.join(" ")));
    }
    if !subs.is_empty() {
        let parts: Vec<String> = subs.iter().map(|s| escape_name(&translate(s))).collect();
        tex.push_str(&format!("_{{{}}}", parts.join(" ")));
    }
    (tex, !supers.is_empty())
}

/// The head of a named function: the command of LaTeX when there is one,
/// the name in `\operatorname` otherwise, whole (`operator_name`).
pub fn operator_name(name: &str) -> String {
    if ACCEPTED_FUNCTIONS.contains(&name) {
        return format!("\\{}", name);
    }
    format!(r"\operatorname{{{}}}", escape_text(name))
}

/// `s` escaped so that LaTeX reads it as plain text, the commands
/// terminated (`escape_text`, SymPy's `latex_escape`).
pub fn escape_text(s: &str) -> String {
    let mut out: String = String::new();
    for c in s.chars() {
        match c {
            '\\' => out.push_str(r"\textbackslash{}"),
            '~' => out.push_str(r"\textasciitilde{}"),
            '^' => out.push_str(r"\textasciicircum{}"),
            '&' => out.push_str(r"\&"),
            '%' => out.push_str(r"\%"),
            '$' => out.push_str(r"\$"),
            '#' => out.push_str(r"\#"),
            '_' => out.push_str(r"\_"),
            '{' => out.push_str(r"\{"),
            '}' => out.push_str(r"\}"),
            _ => out.push(c),
        }
    }
    out
}

/// `s` as monospace text, what an object without a rule prints as (`text`).
pub fn text(s: &str) -> String {
    format!(r"\mathtt{{\text{{{}}}}}", escape_text(s))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn as_the_docstrings() {
        assert_eq!(
            [
                translate("alpha"),
                translate("Gamma"),
                translate("xdot"),
                translate("alphahat"),
                translate("y")
            ],
            [r"\alpha", r"\Gamma", r"\dot{x}", r"\hat{\alpha}", "y"]
        );
        assert_eq!(
            [escape_name("a%b"), escape_name("x_{1}"), escape_name("a{b")],
            [r"a\%b", "x_{1}", r"a\{b"]
        );
        assert_eq!(name_latex("x_1"), ("x_{1}".to_string(), false));
        assert_eq!(name_latex("alpha^2_i"), (r"\alpha^{2}_{i}".to_string(), true));
        assert_eq!(name_latex("x12"), ("x_{12}".to_string(), false));
        assert_eq!(name_latex("a%b"), (r"a\%b".to_string(), false));
        assert_eq!(
            [
                operator_name("sin"),
                operator_name("asinh"),
                operator_name("atan2"),
                operator_name("polar_lift")
            ],
            [
                r"\sin",
                r"\operatorname{asinh}",
                r"\operatorname{atan2}",
                r"\operatorname{polar\_lift}"
            ]
        );
        assert_eq!(
            [escape_text("a\\b"), escape_text("x_1 & 50%")],
            [r"a\textbackslash{}b", r"x\_1 \& 50\%"]
        );
    }
}
