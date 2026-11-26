//! Port of `sympy_extras/printing/pretty/symbology.py`: the characters of
//! the pretty printer, in Unicode and in ASCII. A [`Symbology`] holds the
//! choice between the two (no global state). The tables are generated from
//! the Python module.

// PORTED FROM SYMPY: sympy/printing/pretty/pretty_symbology.py (SymPy 1.14, BSD license; through
// sympy_extras/printing/pretty/symbology.py)

use crate::printing::conventions::split_super_sub;

/// An object drawn at any length: the repeated piece, the end pieces, the
/// middle piece of a brace and the form of length one (`Extensible`).
#[derive(Clone, Copy, Debug)]
pub struct Extensible {
    pub ext: &'static str,
    pub top: &'static str,
    pub bottom: &'static str,
    pub middle: Option<&'static str>,
    pub single: &'static str,
}

impl Extensible {
    /// The object at the given length, one piece per row (or column).
    pub fn pieces(&self, length: usize) -> Vec<&'static str> {
        let mut length: usize = length.max(1);
        if self.middle.is_some() && length % 2 == 0 {
            length += 1;
        }
        if length == 1 {
            return vec![self.single];
        }
        let half: usize = (length - 2) / 2;
        let odd: usize = (length - 2) - 2 * half;
        let middle: &'static str = self.middle.unwrap_or(self.ext);
        let mut out: Vec<&'static str> = vec![self.top];
        out.extend(std::iter::repeat(self.ext).take(half));
        out.extend(std::iter::repeat(middle).take(odd));
        out.extend(std::iter::repeat(self.ext).take(half));
        out.push(self.bottom);
        out
    }
}

/// A stylised letter with what is needed to put indices around it
/// (`Annotated`).
#[derive(Clone, Copy, Debug)]
pub struct Annotated {
    pub size: usize,
    pub top: usize,
    pub bottom: usize,
    pub add: usize,
    pub image: &'static str,
}

#[derive(Clone, Copy, Debug)]
enum ModifierKind {
    /// a combining mark on the middle character
    Accent(&'static str),
    /// a mark after the name
    Suffix(&'static str),
    /// the name between two marks
    Fence(&'static str, &'static str),
}

// GENERATED from sympy_extras/printing/pretty/symbology.py

/// The Greek letters by name (`greek_unicode`).
pub(crate) fn greek_unicode(name: &str) -> Option<&'static str> {
    Some(match name {
        "Alpha" => "\u{391}",
        "Beta" => "\u{392}",
        "Chi" => "\u{3a7}",
        "Delta" => "\u{394}",
        "Epsilon" => "\u{395}",
        "Eta" => "\u{397}",
        "Gamma" => "\u{393}",
        "Iota" => "\u{399}",
        "Kappa" => "\u{39a}",
        "Lambda" => "\u{39b}",
        "Lamda" => "\u{39b}",
        "Mu" => "\u{39c}",
        "Nu" => "\u{39d}",
        "Omega" => "\u{3a9}",
        "Omicron" => "\u{39f}",
        "Phi" => "\u{3a6}",
        "Pi" => "\u{3a0}",
        "Psi" => "\u{3a8}",
        "Rho" => "\u{3a1}",
        "Sigma" => "\u{3a3}",
        "Tau" => "\u{3a4}",
        "Theta" => "\u{398}",
        "Upsilon" => "\u{3a5}",
        "Xi" => "\u{39e}",
        "Zeta" => "\u{396}",
        "alpha" => "\u{3b1}",
        "beta" => "\u{3b2}",
        "chi" => "\u{3c7}",
        "delta" => "\u{3b4}",
        "epsilon" => "\u{3b5}",
        "eta" => "\u{3b7}",
        "gamma" => "\u{3b3}",
        "iota" => "\u{3b9}",
        "kappa" => "\u{3ba}",
        "lambda" => "\u{3bb}",
        "lamda" => "\u{3bb}",
        "mu" => "\u{3bc}",
        "nu" => "\u{3bd}",
        "omega" => "\u{3c9}",
        "omicron" => "\u{3bf}",
        "phi" => "\u{3c6}",
        "pi" => "\u{3c0}",
        "psi" => "\u{3c8}",
        "rho" => "\u{3c1}",
        "sigma" => "\u{3c3}",
        "tau" => "\u{3c4}",
        "theta" => "\u{3b8}",
        "upsilon" => "\u{3c5}",
        "varsigma" => "\u{3c2}",
        "xi" => "\u{3be}",
        "zeta" => "\u{3b6}",
        _ => return None,
    })
}

/// The subscript form of a character or Greek name (`sub`).
pub(crate) fn sub_char(name: &str) -> Option<&'static str> {
    Some(match name {
        "(" => "\u{208d}",
        ")" => "\u{208e}",
        "+" => "\u{208a}",
        "-" => "\u{208b}",
        "0" => "\u{2080}",
        "1" => "\u{2081}",
        "2" => "\u{2082}",
        "3" => "\u{2083}",
        "4" => "\u{2084}",
        "5" => "\u{2085}",
        "6" => "\u{2086}",
        "7" => "\u{2087}",
        "8" => "\u{2088}",
        "9" => "\u{2089}",
        "=" => "\u{208c}",
        "a" => "\u{2090}",
        "beta" => "\u{1d66}",
        "chi" => "\u{1d6a}",
        "e" => "\u{2091}",
        "gamma" => "\u{1d67}",
        "h" => "\u{2095}",
        "i" => "\u{1d62}",
        "k" => "\u{2096}",
        "l" => "\u{2097}",
        "m" => "\u{2098}",
        "n" => "\u{2099}",
        "o" => "\u{2092}",
        "p" => "\u{209a}",
        "phi" => "\u{1d69}",
        "r" => "\u{1d63}",
        "rho" => "\u{1d68}",
        "s" => "\u{209b}",
        "t" => "\u{209c}",
        "u" => "\u{1d64}",
        "v" => "\u{1d65}",
        "x" => "\u{2093}",
        _ => return None,
    })
}

/// The supscript form of a character or Greek name (`sup`).
pub(crate) fn sup_char(name: &str) -> Option<&'static str> {
    Some(match name {
        "(" => "\u{207d}",
        ")" => "\u{207e}",
        "+" => "\u{207a}",
        "-" => "\u{207b}",
        "0" => "\u{2070}",
        "1" => "\u{b9}",
        "2" => "\u{b2}",
        "3" => "\u{b3}",
        "4" => "\u{2074}",
        "5" => "\u{2075}",
        "6" => "\u{2076}",
        "7" => "\u{2077}",
        "8" => "\u{2078}",
        "9" => "\u{2079}",
        "=" => "\u{207c}",
        "i" => "\u{2071}",
        "n" => "\u{207f}",
        _ => return None,
    })
}

fn xobj_unicode(symbol: &str) -> Option<Extensible> {
    Some(match symbol {
        "(" => Extensible {
            ext: "\u{239c}",
            top: "\u{239b}",
            bottom: "\u{239d}",
            middle: None,
            single: "(",
        },
        ")" => Extensible {
            ext: "\u{239f}",
            top: "\u{239e}",
            bottom: "\u{23a0}",
            middle: None,
            single: ")",
        },
        "[" => Extensible {
            ext: "\u{23a2}",
            top: "\u{23a1}",
            bottom: "\u{23a3}",
            middle: None,
            single: "[",
        },
        "]" => Extensible {
            ext: "\u{23a5}",
            top: "\u{23a4}",
            bottom: "\u{23a6}",
            middle: None,
            single: "]",
        },
        "{" => Extensible {
            ext: "\u{23aa}",
            top: "\u{23a7}",
            bottom: "\u{23a9}",
            middle: Some("\u{23a8}"),
            single: "{",
        },
        "}" => Extensible {
            ext: "\u{23aa}",
            top: "\u{23ab}",
            bottom: "\u{23ad}",
            middle: Some("\u{23ac}"),
            single: "}",
        },
        "|" => Extensible {
            ext: "\u{2502}",
            top: "\u{2502}",
            bottom: "\u{2502}",
            middle: None,
            single: "\u{2502}",
        },
        "<" => Extensible {
            ext: "\u{2502}",
            top: "\u{2571}",
            bottom: "\u{2572}",
            middle: None,
            single: "<",
        },
        ">" => Extensible {
            ext: "\u{2502}",
            top: "\u{2572}",
            bottom: "\u{2571}",
            middle: None,
            single: ">",
        },
        "lfloor" => Extensible {
            ext: "\u{23a2}",
            top: "\u{23a2}",
            bottom: "\u{23a3}",
            middle: None,
            single: "\u{230a}",
        },
        "rfloor" => Extensible {
            ext: "\u{23a5}",
            top: "\u{23a5}",
            bottom: "\u{23a6}",
            middle: None,
            single: "\u{230b}",
        },
        "lceil" => Extensible {
            ext: "\u{23a2}",
            top: "\u{23a1}",
            bottom: "\u{23a2}",
            middle: None,
            single: "\u{2308}",
        },
        "rceil" => Extensible {
            ext: "\u{23a5}",
            top: "\u{23a4}",
            bottom: "\u{23a5}",
            middle: None,
            single: "\u{2309}",
        },
        "int" => Extensible {
            ext: "\u{23ae}",
            top: "\u{2320}",
            bottom: "\u{2321}",
            middle: None,
            single: "\u{222b}",
        },
        "-" => Extensible {
            ext: "\u{2500}",
            top: "\u{2500}",
            bottom: "\u{2500}",
            middle: None,
            single: "\u{2500}",
        },
        "_" => Extensible {
            ext: "_",
            top: "_",
            bottom: "_",
            middle: None,
            single: "_",
        },
        "/" => Extensible {
            ext: "\u{2571}",
            top: "\u{2571}",
            bottom: "\u{2571}",
            middle: None,
            single: "\u{2571}",
        },
        "\\" => Extensible {
            ext: "\u{2572}",
            top: "\u{2572}",
            bottom: "\u{2572}",
            middle: None,
            single: "\u{2572}",
        },
        "UpTack" => Extensible {
            ext: "\u{252c}",
            top: "\u{252c}",
            bottom: "\u{252c}",
            middle: None,
            single: "\u{252c}",
        },
        _ => return None,
    })
}

fn xobj_ascii(symbol: &str) -> Option<Extensible> {
    Some(match symbol {
        "(" => Extensible {
            ext: "|",
            top: "/",
            bottom: "\\",
            middle: None,
            single: "(",
        },
        ")" => Extensible {
            ext: "|",
            top: "\\",
            bottom: "/",
            middle: None,
            single: ")",
        },
        "[" => Extensible {
            ext: "[",
            top: "[",
            bottom: "[",
            middle: None,
            single: "[",
        },
        "]" => Extensible {
            ext: "]",
            top: "]",
            bottom: "]",
            middle: None,
            single: "]",
        },
        "{" => Extensible {
            ext: "|",
            top: "/",
            bottom: "\\",
            middle: Some("<"),
            single: "{",
        },
        "}" => Extensible {
            ext: "|",
            top: "\\",
            bottom: "/",
            middle: Some(">"),
            single: "}",
        },
        "|" => Extensible {
            ext: "|",
            top: "|",
            bottom: "|",
            middle: None,
            single: "|",
        },
        "<" => Extensible {
            ext: "|",
            top: "/",
            bottom: "\\",
            middle: None,
            single: "<",
        },
        ">" => Extensible {
            ext: "|",
            top: "\\",
            bottom: "/",
            middle: None,
            single: ">",
        },
        "int" => Extensible {
            ext: " | ",
            top: "  /",
            bottom: "/  ",
            middle: None,
            single: " | ",
        },
        "-" => Extensible {
            ext: "-",
            top: "-",
            bottom: "-",
            middle: None,
            single: "-",
        },
        "_" => Extensible {
            ext: "_",
            top: "_",
            bottom: "_",
            middle: None,
            single: "_",
        },
        "/" => Extensible {
            ext: "/",
            top: "/",
            bottom: "/",
            middle: None,
            single: "/",
        },
        "\\" => Extensible {
            ext: "\\",
            top: "\\",
            bottom: "\\",
            middle: None,
            single: "\\",
        },
        _ => return None,
    })
}

fn xsym_pair(symbol: &str) -> Option<(&'static str, &'static str)> {
    Some(match symbol {
        "==" => ("=", "="),
        "<" => ("<", "<"),
        ">" => (">", ">"),
        "<=" => ("<=", "\u{2264}"),
        ">=" => (">=", "\u{2265}"),
        "!=" => ("!=", "\u{2260}"),
        "*" => ("*", "\u{22c5}"),
        "." => ("*", "\u{2218}"),
        _ => return None,
    })
}

fn atom_unicode(name: &str) -> Option<&'static str> {
    Some(match name {
        "Exp1" => "\u{212f}",
        "Pi" => "\u{3c0}",
        "Infinity" => "\u{221e}",
        "NegativeInfinity" => "-\u{221e}",
        "ImaginaryUnit" => "\u{2148}",
        "EmptySet" => "\u{2205}",
        "Naturals" => "\u{2115}",
        "Naturals0" => "\u{2115}\u{2080}",
        "Integers" => "\u{2124}",
        "Rationals" => "\u{211a}",
        "Reals" => "\u{211d}",
        "Complexes" => "\u{2102}",
        "Universe" => "\u{1d54c}",
        "Differential" => "\u{2146}",
        "Union" => "\u{222a}",
        "ElementOf" => "\u{2208}",
        "SmallElementOf" => "\u{220a}",
        "SymmetricDifference" => "\u{2206}",
        "Intersection" => "\u{2229}",
        "Ring" => "\u{2218}",
        "Multiplication" => "\u{d7}",
        "Dots" => "\u{2026}",
        "SuperscriptPlus" => "\u{207a}",
        "SuperscriptMinus" => "\u{207b}",
        "And" => "\u{2227}",
        "Or" => "\u{2228}",
        "Not" => "\u{ac}",
        "Nor" => "\u{22bd}",
        "Nand" => "\u{22bc}",
        "Xor" => "\u{22bb}",
        "Equiv" => "\u{21d4}",
        "NotEquiv" => "\u{21ce}",
        "Arrow" => "\u{2192}",
        "ArrowFromBar" => "\u{21a6}",
        "NotArrow" => "\u{219b}",
        "Partial" => "\u{2202}",
        "ForAll" => "\u{2200}",
        "Exists" => "\u{2203}",
        _ => return None,
    })
}

fn root_sign(index: i64) -> Option<&'static str> {
    Some(match index {
        2 => "\u{221a}",
        3 => "\u{221b}",
        4 => "\u{221c}",
        _ => return None,
    })
}

fn annotated_unicode(letter: &str) -> Option<Annotated> {
    Some(match letter {
        "F" => Annotated {
            size: 2,
            top: 0,
            bottom: 2,
            add: 0,
            image: "\u{250c}\u{2500}\n\u{251c}\u{2500}\n\u{2575}",
        },
        "G" => Annotated {
            size: 3,
            top: 0,
            bottom: 3,
            add: 1,
            image: "\u{256d}\u{2500}\u{256e}\n\u{2502}\u{2576}\u{2510}\n\u{2570}\u{2500}\u{256f}",
        },
        _ => return None,
    })
}

fn annotated_ascii(letter: &str) -> Option<Annotated> {
    Some(match letter {
        "F" => Annotated {
            size: 3,
            top: 0,
            bottom: 3,
            add: 0,
            image: " _\n|_\n|\n",
        },
        "G" => Annotated {
            size: 3,
            top: 0,
            bottom: 3,
            add: 1,
            image: " __\n/__\n\\_|",
        },
        _ => return None,
    })
}

const MODIFIER_MARKS: [(&str, ModifierKind); 19] = [
    ("mathring", ModifierKind::Accent("\u{30a}")),
    ("ddddot", ModifierKind::Accent("\u{20dc}")),
    ("dddot", ModifierKind::Accent("\u{20db}")),
    ("check", ModifierKind::Accent("\u{30c}")),
    ("breve", ModifierKind::Accent("\u{306}")),
    ("acute", ModifierKind::Accent("\u{301}")),
    ("grave", ModifierKind::Accent("\u{300}")),
    ("tilde", ModifierKind::Accent("\u{303}")),
    ("prime", ModifierKind::Suffix("\u{2032}")),
    ("ddot", ModifierKind::Accent("\u{308}")),
    ("norm", ModifierKind::Fence("\u{2016}", "\u{2016}")),
    ("dot", ModifierKind::Accent("\u{307}")),
    ("hat", ModifierKind::Accent("\u{302}")),
    ("bar", ModifierKind::Accent("\u{305}")),
    ("vec", ModifierKind::Accent("\u{20d7}")),
    ("prm", ModifierKind::Suffix("\u{2032}")),
    ("avg", ModifierKind::Fence("\u{27e8}", "\u{27e9}")),
    ("abs", ModifierKind::Fence("|", "|")),
    ("mag", ModifierKind::Fence("|", "|")),
];

/// The combining marks, which take no column (`_COMBINING`).
fn is_combining(c: char) -> bool {
    matches!(c as u32, 0x300..=0x36e | 0x20d0..=0x20ef)
}

/// The number of columns of `line`: the combining marks take none
/// (`line_width`).
pub fn line_width(line: &str) -> usize {
    line.chars().filter(|c| !is_combining(*c)).count()
}

/// The paddings which centre `wstring` columns in `wtarget` (the odd
/// column goes to the right) (`center_pad`).
pub fn center_pad(wstring: usize, wtarget: usize, fillchar: char) -> (String, String) {
    let delta: usize = wtarget.saturating_sub(wstring);
    let left: usize = delta / 2;
    (
        std::iter::repeat(fillchar).take(left).collect(),
        std::iter::repeat(fillchar).take(delta - left).collect(),
    )
}

/// `string` centred in `width` columns (`center`).
pub fn center(string: &str, width: usize) -> String {
    let (left, right) = center_pad(line_width(string), width, ' ');
    format!("{}{}{}", left, string, right)
}

/// `string` with the combining `accent` on its middle character
/// (`_center_accent`).
fn center_accent(string: &str, accent: &str) -> String {
    let chars: Vec<char> = string.chars().collect();
    let midpoint: usize = (chars.len() / 2 + 1).min(chars.len());
    let head: String = chars[..midpoint].iter().collect();
    let tail: String = chars[midpoint..].iter().collect();
    format!("{}{}{}", head, accent, tail)
}

/// A name without scripts in Unicode: the Greek letters, the modifiers at
/// the end (`_translate`, `translate_name` with these tables).
fn translate(name: &str) -> String {
    if let Some(g) = greek_unicode(name) {
        return g.to_string();
    }
    let lowered: String = name.to_lowercase();
    for (key, kind) in MODIFIER_MARKS.iter() {
        if lowered.ends_with(key) && name.chars().count() > key.chars().count() {
            let head: String = name.chars().take(name.chars().count() - key.chars().count()).collect();
            let inner: String = translate(&head);
            return match kind {
                ModifierKind::Accent(mark) => center_accent(&inner, mark),
                ModifierKind::Suffix(mark) => format!("{}{}", inner, mark),
                ModifierKind::Fence(left, right) => format!("{}{}{}", left, inner, right),
            };
        }
    }
    name.to_string()
}

/// The parts in the script characters, `None` when one of them has no
/// such form (`_scripts`).
fn scripts(parts: &[String], mapping: fn(&str) -> Option<&'static str>) -> Option<Vec<String>> {
    let mut result: Vec<String> = Vec::new();
    for part in parts {
        if let Some(whole) = mapping(part) {
            result.push(whole.to_string());
            continue;
        }
        let mut piece: String = String::new();
        for c in part.chars() {
            piece.push_str(mapping(&c.to_string())?);
        }
        result.push(piece);
    }
    Some(result)
}

/// The characters to draw with: Unicode or ASCII (`Symbology`).
#[derive(Clone, Debug, PartialEq)]
pub struct Symbology {
    pub unicode: bool,
    /// the letter of the imaginary unit in Unicode (`'i'` or `'j'`)
    pub imaginary_unit: String,
}

impl Symbology {
    pub fn new(unicode: bool) -> Symbology {
        Symbology {
            unicode,
            imaginary_unit: "i".to_string(),
        }
    }

    fn table(&self, symbol: &str) -> Extensible {
        let found = if self.unicode {
            xobj_unicode(symbol)
        } else {
            xobj_ascii(symbol)
        };
        found.unwrap_or_else(|| panic!("no extensible object {:?}", symbol))
    }

    /// The pieces of the extensible object `symbol` at `length` (`xobj`).
    pub fn xobj(&self, symbol: &str, length: usize) -> Vec<&'static str> {
        self.table(symbol).pieces(length)
    }

    /// The object drawn vertically, its rows separated by newlines (`vobj`).
    pub fn vobj(&self, symbol: &str, height: usize) -> String {
        self.xobj(symbol, height).join("\n")
    }

    /// The object drawn horizontally (`hobj`).
    pub fn hobj(&self, symbol: &str, width: usize) -> String {
        self.xobj(symbol, width).concat()
    }

    /// The operator `symbol` (`xsym`).
    pub fn xsym(&self, symbol: &str) -> &'static str {
        let (ascii, unicode) = xsym_pair(symbol).unwrap_or_else(|| panic!("no operator {:?}", symbol));
        if self.unicode {
            unicode
        } else {
            ascii
        }
    }

    /// The Unicode character of the atom `name`, `default` in ASCII (`atom`).
    pub fn atom(&self, name: &str, default: &str) -> String {
        if !self.unicode {
            return default.to_string();
        }
        if name == "ImaginaryUnit" && self.imaginary_unit == "j" {
            return "\u{2149}".to_string();
        }
        atom_unicode(name).unwrap_or(default).to_string()
    }

    /// The Greek letter `name`, `default` in ASCII (`greek`).
    pub fn greek(&self, name: &str, default: &str) -> String {
        if self.unicode {
            greek_unicode(name).unwrap_or(default).to_string()
        } else {
            default.to_string()
        }
    }

    /// The one-character root sign of the given index, if there is one (`root`).
    pub fn root(&self, index: i64) -> Option<&'static str> {
        if self.unicode {
            root_sign(index)
        } else {
            None
        }
    }

    /// The stylised `F` or `G` (`annotated`).
    pub fn annotated(&self, letter: &str) -> Annotated {
        let found = if self.unicode {
            annotated_unicode(letter)
        } else {
            annotated_ascii(letter)
        };
        found.unwrap_or_else(|| panic!("no annotated letter {:?}", letter))
    }

    /// Whether every character of `text` has a Unicode subscript (`subscriptable`).
    pub fn subscriptable(&self, text: &str) -> bool {
        self.unicode && !text.is_empty() && text.chars().all(|c| sub_char(&c.to_string()).is_some())
    }

    /// The printed form of a symbol name (`symbol`).
    pub fn symbol(&self, name: &str) -> String {
        if !self.unicode {
            return name.to_string();
        }
        let (base, sups, subs) = split_super_sub(name);
        if sups.iter().any(String::is_empty) || subs.iter().any(String::is_empty) || base.is_empty() {
            // x_ is not x: SymPy drops the empty mark
            return name.to_string();
        }
        let mut base: String = translate(&base);
        let pretty_sups: Option<Vec<String>> = scripts(&sups, sup_char);
        let pretty_subs: Option<Vec<String>> = if pretty_sups.is_some() {
            scripts(&subs, sub_char)
        } else {
            None
        };
        match (pretty_sups, pretty_subs) {
            (Some(a), Some(b)) => format!("{}{}{}", base, a.join(" "), b.join(" ")),
            _ => {
                if !subs.is_empty() {
                    let parts: Vec<String> = subs.iter().map(|s| translate(s)).collect();
                    base.push('_');
                    base.push_str(&parts.join("_"));
                }
                if !sups.is_empty() {
                    let parts: Vec<String> = sups.iter().map(|s| translate(s)).collect();
                    base.push_str("__");
                    base.push_str(&parts.join("__"));
                }
                base
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn as_the_docstrings() {
        let (ascii, uni) = (Symbology::new(false), Symbology::new(true));
        assert_eq!(ascii.xsym("<="), "<=");
        assert_eq!(ascii.symbol("alpha_1"), "alpha_1");
        assert_eq!(ascii.hobj("-", 3), "---");
        assert_eq!(uni.xsym("<="), "\u{2264}");
        assert_eq!(uni.symbol("alpha_1"), "\u{3b1}\u{2081}");
        assert_eq!(line_width("x\u{307}"), 1);
    }
}
