//! Port of `sympy_extras/printing/pretty/layout.py`: two-dimensional
//! pictures, the output of the pretty printer. A [`Picture`] is an
//! immutable block of rows of equal width with a baseline; the binding of
//! a picture is not a field of it but of the typed result of a rule.

// PORTED FROM SYMPY: sympy/printing/pretty/stringpict.py (stringPict) (SymPy 1.14, BSD license; through
// sympy_extras/printing/pretty/layout.py)

use super::symbology::{center, line_width, Symbology};
use crate::printing::framework::{Context, PrintResult};

/// An immutable block of text: rows of equal width, and the index of the
/// row which is the baseline (`Picture`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Picture {
    pub rows: Vec<String>,
    /// the baseline row; negative as in Python (the rows then stand
    /// below the baseline row of the pictures beside it)
    pub baseline: i64,
}

impl Picture {
    pub fn new(rows: Vec<String>, baseline: i64) -> Picture {
        let lines: Vec<String> = if rows.is_empty() { vec![String::new()] } else { rows };
        let width: usize = lines.iter().map(|l| line_width(l)).max().unwrap_or(0);
        Picture {
            rows: lines.iter().map(|l| center(l, width)).collect(),
            baseline,
        }
    }

    /// The picture of a string; the lines of a multiline string are centred
    /// (`text`).
    pub fn text(s: &str, baseline: i64) -> Picture {
        // Python's splitlines: no final empty line for a trailing newline
        let lines: Vec<String> = s.lines().map(str::to_string).collect();
        Picture::new(lines, baseline)
    }

    /// A one-row picture of a string.
    pub fn of(s: &str) -> Picture {
        Picture::text(s, 0)
    }

    /// A block of spaces (`blank`).
    pub fn blank(width: usize, height: usize) -> Picture {
        Picture::new(vec![" ".repeat(width); height], 0)
    }

    pub fn height(&self) -> usize {
        self.rows.len()
    }

    pub fn width(&self) -> usize {
        line_width(&self.rows[0])
    }

    pub fn with_baseline(&self, baseline: i64) -> Picture {
        Picture::new(self.rows.clone(), baseline)
    }

    /// The pictures next to each other, their baselines aligned (`beside`).
    pub fn beside(pictures: &[&Picture]) -> Picture {
        if pictures.is_empty() {
            return Picture::new(vec![String::new()], 0);
        }
        // Python's arithmetic, a negative baseline included
        let baseline: i64 = pictures.iter().map(|p| p.baseline).max().unwrap_or(0);
        let below: i64 = pictures
            .iter()
            .map(|p| p.height() as i64 - p.baseline)
            .max()
            .unwrap_or(0);
        let height: i64 = baseline + below;
        let mut columns: Vec<Vec<String>> = Vec::new();
        for p in pictures {
            let empty: String = " ".repeat(p.width());
            let top: i64 = baseline - p.baseline;
            let mut column: Vec<String> = vec![empty.clone(); top.max(0) as usize];
            column.extend(p.rows.iter().cloned());
            let rest: i64 = height - p.height() as i64 - top;
            column.extend(std::iter::repeat(empty.clone()).take(rest.max(0) as usize));
            columns.push(column);
        }
        let rows: Vec<String> = (0..height.max(0) as usize)
            .map(|i| {
                columns
                    .iter()
                    .map(|c| c.get(i).map(String::as_str).unwrap_or(""))
                    .collect::<String>()
            })
            .collect();
        Picture::new(rows, baseline)
    }

    /// The pictures put at the right of this one (`right`).
    pub fn right(&self, pictures: &[&Picture]) -> Picture {
        let mut all: Vec<&Picture> = vec![self];
        all.extend(pictures.iter().copied());
        Picture::beside(&all)
    }

    /// The pictures (from left to right) put at the left of this one (`left`).
    pub fn left(&self, pictures: &[&Picture]) -> Picture {
        let mut all: Vec<&Picture> = pictures.to_vec();
        all.push(self);
        Picture::beside(&all)
    }

    /// This picture followed by a string on the baseline (`rtext`).
    pub fn rtext(&self, s: &str) -> Picture {
        Picture::beside(&[self, &Picture::of(s)])
    }

    /// This picture preceded by a string on the baseline (`ltext`).
    pub fn ltext(&self, s: &str) -> Picture {
        Picture::beside(&[&Picture::of(s), self])
    }

    /// The rows of the pictures on top of each other, centred (`stacked`).
    pub fn stacked(pictures: &[&Picture]) -> Vec<String> {
        let width: usize = pictures.iter().map(|p| p.width()).max().unwrap_or(0);
        pictures
            .iter()
            .flat_map(|p| p.rows.iter().map(move |r| center(r, width)))
            .collect()
    }

    /// The pictures put under this one, which keeps the baseline (`below`).
    pub fn below(&self, pictures: &[&Picture]) -> Picture {
        let mut all: Vec<&Picture> = vec![self];
        all.extend(pictures.iter().copied());
        Picture::new(Picture::stacked(&all), self.baseline)
    }

    /// The pictures (from top to bottom) put over this one, which keeps the
    /// baseline (`above`).
    pub fn above(&self, pictures: &[&Picture]) -> Picture {
        let mut all: Vec<&Picture> = pictures.to_vec();
        all.push(self);
        let rows: Vec<String> = Picture::stacked(&all);
        let baseline: i64 = rows.len() as i64 - self.height() as i64 + self.baseline;
        Picture::new(rows, baseline)
    }

    /// This picture between delimiters of its height (an empty string is no
    /// delimiter); `one_row_in_ascii` keeps them one row high in ASCII
    /// (`parens`).
    pub fn parens(&self, symbology: &Symbology, left: &str, right: &str, one_row_in_ascii: bool) -> Picture {
        let (mut height, mut baseline): (usize, i64) = (self.height(), self.baseline);
        if one_row_in_ascii && !symbology.unicode {
            height = 1;
            baseline = 0;
        }
        let mut result: Picture = self.clone();
        if !left.is_empty() {
            result = Picture::text(&symbology.vobj(left, height), baseline).right(&[&result]);
        }
        if !right.is_empty() {
            result = result.right(&[&Picture::text(&symbology.vobj(right, height), baseline)]);
        }
        result
    }

    /// The picture between round brackets.
    pub fn round(&self, symbology: &Symbology) -> Picture {
        self.parens(symbology, "(", ")", false)
    }

    /// The picture as a string, cut into screens of `num_columns` columns
    /// when it is wider and `wrap_line` is set (`render`).
    pub fn render(&self, wrap_line: bool, num_columns: Option<usize>, unicode: bool) -> String {
        if !wrap_line {
            return self.rows.join("\n");
        }
        let columns: usize = match num_columns {
            Some(c) if c > 0 => c,
            _ => 80,
        };
        if self.width() <= columns {
            return self.rows.join("\n");
        }
        let (start, end) = if unicode {
            ("\u{21aa} ", " \u{21aa}")
        } else {
            ("> ", " >")
        };
        let chunks: Vec<Vec<String>> = self.rows.iter().map(|r| chunks(r, columns, start, end)).collect();
        let screens: usize = chunks.iter().map(Vec::len).min().unwrap_or(0);
        (0..screens)
            .map(|i| chunks.iter().map(|c| c[i].as_str()).collect::<Vec<&str>>().join("\n"))
            .collect::<Vec<String>>()
            .join("\n\n")
    }
}

impl std::fmt::Display for Picture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.rows.join("\n"))
    }
}

/// The consecutive pieces of `line` which fit `columns` columns (`_chunks`).
fn chunks(line: &str, columns: usize, overflow_start: &str, overflow_end: &str) -> Vec<String> {
    let chars: Vec<char> = line.chars().collect();
    let mut out: Vec<String> = Vec::new();
    let mut prefix: String = String::new();
    let mut width: usize = line_width(&format!("{}{}", prefix, overflow_end));
    let mut start: usize = 0;
    for (i, c) in chars.iter().enumerate() {
        let w: usize = line_width(&c.to_string());
        // flush only when the character overflows: a combining mark can
        // still be appended at the full width
        if width + w > columns {
            let piece: String = chars[start..i].iter().collect();
            out.push(format!("{}{}{}", prefix, piece, overflow_end));
            prefix = overflow_start.to_string();
            width = line_width(&format!("{}{}", prefix, overflow_end));
            start = i;
        }
        width += w;
    }
    let piece: String = chars[start..].iter().collect();
    out.push(format!("{}{}", prefix, piece));
    out
}

/// What a rule of the pretty printer returns (`PrettyResult`).
pub type PrettyResult = PrintResult<Picture>;

/// What a parent hands a child (`PrettyContext`).
pub type PrettyContext = Context<Picture>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fraction_beside_a_sum() {
        let half: Picture = Picture::of("1")
            .below(&[&Picture::of("-"), &Picture::of("2")])
            .with_baseline(1);
        let total: Picture = Picture::beside(&[&Picture::of("10"), &Picture::of(" + "), &half]);
        assert_eq!(total.to_string(), "     1\n10 + -\n     2");
    }
}
