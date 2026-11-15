//! Small reusable pieces: cards, pills, big buttons and number formatting.

use crate::theme::Palette;
use crate::theme::Semibold;
use egui::{
    Align, Button, Color32, CornerRadius, Frame, Layout, Margin, RichText, Stroke, Ui, vec2,
};

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
            ui.ctx().request_repaint();
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

pub fn fmt_gb(gb: f64) -> String {
    if gb >= 1000.0 {
        format!("{:.2} TB", gb / 1000.0)
    } else if gb >= 100.0 {
        format!("{gb:.0} GB")
    } else {
        format!("{gb:.1} GB")
    }
}

pub fn fmt_rate(bytes_per_s: f64) -> String {
    if bytes_per_s >= 1e6 {
        format!("{:.1} MB/s", bytes_per_s / 1e6)
    } else {
        format!("{:.0} kB/s", bytes_per_s / 1e3)
    }
}

pub fn fmt_duration(secs: f64, ro: bool) -> String {
    let s = secs.max(0.0) as u64;
    let (h, m) = (s / 3600, (s % 3600) / 60);
    let (hu, mu, su) = if ro {
        ("h", "min", "s")
    } else {
        ("h", "min", "s")
    };
    if h > 0 {
        format!("{h} {hu} {m} {mu}")
    } else if m > 0 {
        format!("{m} {mu}")
    } else {
        format!("{s} {su}")
    }
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
}
