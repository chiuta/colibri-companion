//! Small reusable pieces: cards, pills, big buttons and number formatting.

use crate::theme::Palette;
use crate::theme::Semibold;
use egui::{
    Align, Button, Color32, CornerRadius, Frame, Layout, Margin, RichText, Stroke, Ui, vec2,
};

/// A spinning arc like egui's own `Spinner`, but redrawn about 30 times a second instead of at
/// every screen refresh (egui's asks for a repaint on every frame, for as long as it is shown).
pub fn spinner(ui: &mut Ui) {
    let size = ui.style().spacing.interact_size.y;
    spinner_sized(ui, size);
}

pub fn spinner_sized(ui: &mut Ui, size: f32) {
    let (rect, _) = ui.allocate_exact_size(vec2(size, size), egui::Sense::hover());
    if !ui.is_rect_visible(rect) {
        return;
    }
    ui.ctx()
        .request_repaint_after(std::time::Duration::from_millis(33));
    let radius = (rect.height() / 2.0) - 2.0;
    let n = (radius.round() as u32).clamp(8, 128);
    let time = ui.input(|i| i.time);
    let start = time * std::f64::consts::TAU;
    let end = start + 240f64.to_radians() * time.sin();
    let points: Vec<egui::Pos2> = (0..=n)
        .map(|i| {
            let a = start + (end - start) * f64::from(i) / f64::from(n);
            let (sin, cos) = a.sin_cos();
            rect.center() + radius * vec2(cos as f32, sin as f32)
        })
        .collect();
    ui.painter().add(egui::Shape::line(
        points,
        Stroke::new(3.0, ui.visuals().strong_text_color()),
    ));
}

pub fn card<R>(ui: &mut Ui, p: &Palette, add: impl FnOnce(&mut Ui) -> R) -> R {
    Frame::new()
        .fill(p.card)
        .stroke(Stroke::new(1.0, p.border))
        .corner_radius(CornerRadius::same(12))
        .inner_margin(Margin::same(16))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            add(ui)
        })
        .inner
}

pub fn tinted<R>(ui: &mut Ui, fill: Color32, stroke: Color32, add: impl FnOnce(&mut Ui) -> R) -> R {
    Frame::new()
        .fill(fill)
        .stroke(Stroke::new(1.0, stroke))
        .corner_radius(CornerRadius::same(10))
        .inner_margin(Margin::symmetric(14, 10))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            add(ui)
        })
        .inner
}

pub fn pill(ui: &mut Ui, text: &str, fg: Color32, bg: Color32) {
    Frame::new()
        .fill(bg)
        .corner_radius(CornerRadius::same(20))
        .inner_margin(Margin::symmetric(10, 3))
        .show(ui, |ui| {
            ui.label(RichText::new(text).color(fg).size(13.0).semibold());
        });
}

/// The main call to action on a screen.
pub fn primary_button(ui: &mut Ui, p: &Palette, text: &str) -> egui::Response {
    ui.add(
        Button::new(RichText::new(text).color(p.on_accent).size(17.0).semibold())
            .fill(p.accent)
            .stroke(Stroke::NONE)
            .corner_radius(CornerRadius::same(10))
            .min_size(vec2(180.0, 44.0)),
    )
}

pub fn secondary_button(ui: &mut Ui, text: &str) -> egui::Response {
    ui.add(Button::new(RichText::new(text).size(15.0)).min_size(vec2(0.0, 38.0)))
}

/// A progress bar that stays readable in every theme. `None` shows a moving
/// "working on it" bar for steps whose length is not known.
pub fn progress_bar(ui: &mut Ui, p: &Palette, fraction: Option<f32>, label: &str) {
    let h = 26.0;
    let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), h), egui::Sense::hover());
    let r = CornerRadius::same(13);
    ui.painter().rect_filled(rect, r, p.panel);
    ui.painter().rect_stroke(
        rect,
        r,
        Stroke::new(1.0, p.border),
        egui::StrokeKind::Inside,
    );
    match fraction {
        Some(f) => {
            let mut fill = rect;
            fill.set_width((rect.width() * f.clamp(0.0, 1.0)).max(h.min(rect.width() * 0.04)));
            ui.painter().rect_filled(fill, r, p.accent);
        }
        None => {
            let t = ui.input(|i| i.time) as f32;
            let w = rect.width() * 0.25;
            let x = rect.left() + ((t * 0.7).fract()) * (rect.width() - w);
            let bar = egui::Rect::from_min_size(egui::pos2(x, rect.top()), vec2(w, h));
            ui.painter().rect_filled(bar, r, p.accent);
            // the bar slides, so it needs redraws; about 30 a second look smooth and cost half of
            // what redrawing at every screen refresh does, for as long as a model loads
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(33));
        }
    }
    if !label.is_empty() {
        let left = fraction.map(|f| rect.width() * f > 70.0).unwrap_or(false);
        let color = if left { p.on_accent } else { p.text };
        let pos = if left {
            egui::pos2(rect.left() + 12.0, rect.center().y)
        } else {
            rect.center()
        };
        let align = if left {
            egui::Align2::LEFT_CENTER
        } else {
            egui::Align2::CENTER_CENTER
        };
        ui.painter()
            .text(pos, align, label, egui::FontId::proportional(14.0), color);
    }
}

pub fn heading(ui: &mut Ui, text: &str) {
    ui.label(RichText::new(text).size(26.0).semibold());
}

pub fn muted(ui: &mut Ui, p: &Palette, text: &str) {
    ui.label(RichText::new(text).color(p.muted));
}

/// A centered column no wider than `width`, so text stays comfortable to read.
pub fn column<R>(ui: &mut Ui, width: f32, add: impl FnOnce(&mut Ui) -> R) -> R {
    let avail = ui.available_width();
    let w = width.min(avail);
    let pad = ((avail - w) / 2.0).max(0.0);
    ui.horizontal_top(|ui| {
        ui.add_space(pad);
        ui.vertical(|ui| {
            ui.set_width(w);
            add(ui)
        })
        .inner
    })
    .inner
}

pub fn right_to_left<R>(ui: &mut Ui, add: impl FnOnce(&mut Ui) -> R) -> R {
    ui.with_layout(Layout::right_to_left(Align::Center), add)
        .inner
}

// ------------------------------------------------------------------ formatting

/// The language the numbers are written in. Set once per frame by the window; until then (and in
/// tests) numbers keep the plain English form.
static NUMBER_LANG: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(u8::MAX);

pub fn set_number_lang(l: companion_core::Lang) {
    use companion_core::Lang::*;
    let n = match l {
        Ro => 0,
        En => 1,
        Fr => 2,
        It => 3,
        Es => 4,
        Pt => 5,
        De => 6,
        Hu => 7,
    };
    NUMBER_LANG.store(n, std::sync::atomic::Ordering::Relaxed);
}

/// A number with units as English writes it ("19.4 GB") in the reader's way: a decimal comma in
/// every language except English, and the French octet units (Go, Mo, To, ko). Only a dot
/// between two digits is touched, so text around the number is left alone.
pub fn localise_number(s: &str, lang_index: u8) -> String {
    if lang_index == 1 || lang_index > 7 {
        return s.to_string();
    }
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    for (i, c) in chars.iter().enumerate() {
        let between_digits = *c == '.'
            && i > 0
            && chars[i - 1].is_ascii_digit()
            && chars.get(i + 1).is_some_and(char::is_ascii_digit);
        out.push(if between_digits { ',' } else { *c });
    }
    if lang_index == 2 {
        for (from, to) in [
            (" GB", " Go"),
            (" MB", " Mo"),
            (" TB", " To"),
            (" kB", " ko"),
        ] {
            out = out.replace(from, to);
        }
    }
    out
}

/// [`localise_number`] for the language set with [`set_number_lang`].
pub fn loc(s: String) -> String {
    localise_number(&s, NUMBER_LANG.load(std::sync::atomic::Ordering::Relaxed))
}

pub fn fmt_gb(gb: f64) -> String {
    loc(if gb >= 1000.0 {
        format!("{:.2} TB", gb / 1000.0)
    } else if gb >= 100.0 {
        format!("{gb:.0} GB")
    } else {
        format!("{gb:.1} GB")
    })
}

pub fn fmt_rate(bytes_per_s: f64) -> String {
    loc(if bytes_per_s >= 1e6 {
        format!("{:.1} MB/s", bytes_per_s / 1e6)
    } else {
        format!("{:.0} kB/s", bytes_per_s / 1e3)
    })
}

pub fn fmt_duration(secs: f64, ro: bool) -> String {
    let s = secs.max(0.0) as u64;
    let (h, m) = (s / 3600, (s % 3600) / 60);
    let _ = ro; // the unit symbols are the same in every language the app speaks
    let (hu, mu, su) = ("h", "min", "s");
    if h > 0 {
        format!("{h} {hu} {m} {mu}")
    } else if m > 0 {
        format!("{m} {mu}")
    } else {
        format!("{s} {su}")
    }
}

/// A window size that never exceeds the screen: small laptops and large text sizes still fit.
pub fn fit(ctx: &egui::Context, w: f32, h: f32) -> egui::Vec2 {
    let r = ctx.content_rect();
    egui::vec2(w.min(r.width() - 24.0), h.min(r.height() - 56.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats() {
        assert_eq!(fmt_gb(19.44), "19.4 GB");
        assert_eq!(fmt_gb(429.3), "429 GB");
        assert_eq!(fmt_gb(1561.0), "1.56 TB");
        assert_eq!(fmt_rate(25_000_000.0), "25.0 MB/s");
        assert_eq!(fmt_rate(800_000.0), "800 kB/s");
        assert_eq!(fmt_duration(30.0, true), "30 s");
        assert_eq!(fmt_duration(600.0, true), "10 min");
        assert_eq!(fmt_duration(7500.0, false), "2 h 5 min");
    }

    #[test]
    fn numbers_are_written_the_readers_way() {
        // 0 = Romanian, 1 = English, 2 = French, 6 = German
        assert_eq!(localise_number("19.4 GB", 1), "19.4 GB");
        assert_eq!(localise_number("19.4 GB", 0), "19,4 GB");
        assert_eq!(localise_number("19.4 GB", 2), "19,4 Go");
        assert_eq!(
            localise_number("25.0 MB/s · 1.56 TB · 800 kB/s", 2),
            "25,0 Mo/s · 1,56 To · 800 ko/s"
        );
        assert_eq!(localise_number("19.4 GB", 6), "19,4 GB");
        // text around a number, versions and sentence dots stay as they are
        assert_eq!(
            localise_number("v0.1.0 ready. Next", 6),
            "v0,1,0 ready. Next"
        );
        assert_eq!(localise_number("Done. 3 files", 6), "Done. 3 files");
        assert_eq!(localise_number("19.4 GB", 200), "19.4 GB");
    }
}
