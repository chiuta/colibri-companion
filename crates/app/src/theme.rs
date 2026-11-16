//! Colours, spacing and sizes. Three looks: light, dark and a high-contrast one
//! for low vision. Targets are large and spaced out on purpose.

use companion_core::Theme;
use companion_core::access::Access;
use egui::{Color32, CornerRadius, Stroke, Theme as EguiTheme, Visuals, vec2};

#[derive(Clone, Copy)]
pub struct Palette {
    pub bg: Color32,
    pub panel: Color32,
    pub card: Color32,
    pub text: Color32,
    pub muted: Color32,
    pub accent: Color32,
    pub on_accent: Color32,
    pub accent_soft: Color32,
    pub ok: Color32,
    pub warn: Color32,
    pub warn_soft: Color32,
    pub bad: Color32,
    pub bad_soft: Color32,
    pub border: Color32,
    pub bubble_user: Color32,
}

/// With `colour_blind`, good and bad are blue and orange instead of green and red, which the
/// common kinds of colour blindness keep apart.
pub fn palette_for(theme: Theme, colour_blind: bool) -> Palette {
    let mut p = base_palette(theme);
    if colour_blind {
        let c = Color32::from_rgb;
        match theme {
            Theme::Light => {
                p.ok = c(0, 60, 130);
                p.warn = c(110, 82, 0);
                p.bad = c(180, 60, 0);
                p.bad_soft = c(253, 236, 224);
            }
            Theme::Dark => {
                p.ok = c(150, 210, 255);
                p.warn = c(240, 228, 66);
                p.bad = c(235, 120, 60);
                p.bad_soft = c(70, 38, 20);
            }
            Theme::HighContrast => {
                p.ok = c(120, 200, 255);
                p.warn = c(255, 240, 80);
                p.bad = c(255, 160, 90);
                p.bad_soft = c(50, 20, 0);
            }
        }
    }
    p
}

fn base_palette(theme: Theme) -> Palette {
    match theme {
        Theme::Light => Palette {
            bg: Color32::from_rgb(250, 250, 247),
            panel: Color32::from_rgb(241, 243, 239),
            card: Color32::WHITE,
            text: Color32::from_rgb(28, 32, 30),
            muted: Color32::from_rgb(98, 106, 102),
            accent: Color32::from_rgb(12, 122, 105),
            on_accent: Color32::WHITE,
            accent_soft: Color32::from_rgb(222, 242, 236),
            ok: Color32::from_rgb(20, 125, 58),
            warn: Color32::from_rgb(150, 92, 0),
            warn_soft: Color32::from_rgb(255, 243, 214),
            bad: Color32::from_rgb(171, 30, 30),
            bad_soft: Color32::from_rgb(253, 232, 232),
            border: Color32::from_rgb(214, 219, 214),
            bubble_user: Color32::from_rgb(222, 242, 236),
        },
        Theme::Dark => Palette {
            bg: Color32::from_rgb(22, 25, 24),
            panel: Color32::from_rgb(30, 34, 33),
            card: Color32::from_rgb(36, 41, 40),
            text: Color32::from_rgb(228, 233, 230),
            muted: Color32::from_rgb(154, 164, 159),
            accent: Color32::from_rgb(70, 196, 174),
            on_accent: Color32::from_rgb(8, 30, 26),
            accent_soft: Color32::from_rgb(28, 60, 54),
            ok: Color32::from_rgb(98, 210, 138),
            warn: Color32::from_rgb(240, 186, 90),
            warn_soft: Color32::from_rgb(64, 50, 22),
            bad: Color32::from_rgb(255, 140, 140),
            bad_soft: Color32::from_rgb(70, 30, 30),
            border: Color32::from_rgb(58, 66, 63),
            bubble_user: Color32::from_rgb(30, 64, 58),
        },
        Theme::HighContrast => Palette {
            bg: Color32::BLACK,
            panel: Color32::from_rgb(12, 12, 12),
            card: Color32::from_rgb(8, 8, 8),
            text: Color32::WHITE,
            muted: Color32::from_rgb(220, 220, 220),
            accent: Color32::from_rgb(255, 221, 0),
            on_accent: Color32::BLACK,
            accent_soft: Color32::from_rgb(40, 36, 0),
            ok: Color32::from_rgb(110, 255, 140),
            warn: Color32::from_rgb(255, 200, 80),
            warn_soft: Color32::from_rgb(40, 28, 0),
            bad: Color32::from_rgb(255, 130, 130),
            bad_soft: Color32::from_rgb(45, 0, 0),
            border: Color32::WHITE,
            bubble_user: Color32::from_rgb(30, 30, 30),
        },
    }
}

pub fn apply(ctx: &egui::Context, theme: Theme, zoom: f32, a: &Access) {
    let p = palette_for(theme, a.colour_blind);
    let dark = theme != Theme::Light;
    let mut v = if dark {
        Visuals::dark()
    } else {
        Visuals::light()
    };
    let radius = CornerRadius::same(8);
    v.panel_fill = p.bg;
    v.window_fill = p.card;
    v.extreme_bg_color = p.card;
    v.faint_bg_color = p.panel;
    v.hyperlink_color = p.accent;
    v.code_bg_color = p.panel;
    v.override_text_color = Some(p.text);
    v.selection.bg_fill = if theme == Theme::HighContrast {
        Color32::from_rgb(60, 60, 0)
    } else {
        p.accent_soft
    };
    v.selection.stroke = Stroke::new(1.5, p.accent);
    v.window_stroke = Stroke::new(
        if theme == Theme::HighContrast {
            2.0
        } else {
            1.0
        },
        p.border,
    );
    v.window_corner_radius = CornerRadius::same(12);
    let line = if theme == Theme::HighContrast {
        1.5
    } else {
        1.0
    };
    for w in [
        &mut v.widgets.noninteractive,
        &mut v.widgets.inactive,
        &mut v.widgets.hovered,
        &mut v.widgets.active,
        &mut v.widgets.open,
    ] {
        w.corner_radius = radius;
    }
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, p.border);
    v.widgets.noninteractive.fg_stroke = Stroke::new(1.0, p.text);
    v.widgets.inactive.bg_fill = p.panel;
    v.widgets.inactive.weak_bg_fill = p.panel;
    v.widgets.inactive.bg_stroke = Stroke::new(line, p.border);
    v.widgets.inactive.fg_stroke = Stroke::new(1.0, p.text);
    v.widgets.hovered.bg_fill = p.accent_soft;
    v.widgets.hovered.weak_bg_fill = p.accent_soft;
    v.widgets.hovered.bg_stroke = Stroke::new(1.5, p.accent);
    v.widgets.hovered.fg_stroke = Stroke::new(1.5, p.text);
    v.widgets.active.bg_fill = p.accent_soft;
    v.widgets.active.weak_bg_fill = p.accent_soft;
    v.widgets.active.bg_stroke = Stroke::new(2.0, p.accent);
    ctx.set_theme(if dark {
        EguiTheme::Dark
    } else {
        EguiTheme::Light
    });
    ctx.set_visuals_of(
        if dark {
            EguiTheme::Dark
        } else {
            EguiTheme::Light
        },
        v,
    );
    let sp = a.spacing.clamp(1.0, 2.0);
    let big = a.targets.height();
    let focus = if a.strong_focus { 3.5 } else { 1.5 };
    let readable = a.readable_font;
    ctx.global_style_mut(|s| {
        s.spacing.item_spacing = vec2(10.0 * sp.min(1.4), 9.0 * sp);
        s.spacing.button_padding = vec2(14.0, 8.0 + (big - 34.0) / 4.0);
        s.spacing.interact_size.y = big;
        s.spacing.interact_size.x = (big * 1.3).max(40.0);
        s.spacing.icon_width = (big * 0.55).max(18.0);
        s.visuals.selection.stroke = Stroke::new(focus, p.accent);
        s.animation_time = if a.reduce_motion { 0.0 } else { 1.0 / 12.0 };
        let _ = readable;
        s.spacing.text_edit_width = 320.0;
        s.spacing.scroll.floating = true;
        s.interaction.tooltip_delay = 0.4;
    });
    ctx.set_zoom_factor(zoom.clamp(0.8, 3.0));
    ctx.data_mut(|d| d.insert_temp(egui::Id::unique("reduce-motion"), a.reduce_motion));
    let key = egui::Id::unique("font-readable");
    if ctx.data(|d| d.get_temp::<bool>(key)) != Some(a.readable_font) {
        install_fonts(ctx, a.readable_font);
        ctx.data_mut(|d| d.insert_temp(key, a.readable_font));
    }
}

/// A spinner, or just three dots when the person asked for no movement.
pub fn busy(ui: &mut egui::Ui) {
    let still = ui
        .ctx()
        .data(|d| d.get_temp::<bool>(egui::Id::unique("reduce-motion")))
        .unwrap_or(false);
    if still {
        ui.label("…");
    } else {
        ui.spinner();
    }
}

/// Name of the font family used for emphasis (egui has no synthetic bold).
pub const SEMIBOLD: &str = "semibold";

/// Inter for text and JetBrains Mono for code, both under the SIL OFL 1.1 (see licenses/).
/// The built-in fonts stay as fallbacks for symbols and other scripts.
pub fn install_fonts(ctx: &egui::Context, readable: bool) {
    use egui::{FontData, FontDefinitions, FontFamily};
    use std::sync::Arc;
    let mut f = FontDefinitions::default();
    let add = |f: &mut FontDefinitions, name: &str, bytes: &'static [u8]| {
        f.font_data
            .insert(name.to_owned(), Arc::new(FontData::from_static(bytes)));
    };
    add(
        &mut f,
        "inter",
        include_bytes!("../assets/fonts/Inter-Regular.ttf"),
    );
    add(
        &mut f,
        "inter-semibold",
        include_bytes!("../assets/fonts/Inter-SemiBold.ttf"),
    );
    add(
        &mut f,
        "atkinson",
        include_bytes!("../assets/fonts/AtkinsonHyperlegibleNext-Regular.ttf"),
    );
    add(
        &mut f,
        "jbmono",
        include_bytes!("../assets/fonts/JetBrainsMono-Regular.ttf"),
    );
    let prop = f
        .families
        .entry(FontFamily::Proportional)
        .or_default()
        .clone();
    f.families
        .get_mut(&FontFamily::Proportional)
        .unwrap()
        .insert(0, if readable { "atkinson" } else { "inter" }.into());
    f.families
        .get_mut(&FontFamily::Monospace)
        .unwrap()
        .insert(0, "jbmono".into());
    let mut bold = vec![
        if readable {
            "atkinson"
        } else {
            "inter-semibold"
        }
        .to_owned(),
    ];
    bold.extend(prop);
    f.families.insert(FontFamily::Name(SEMIBOLD.into()), bold);
    ctx.set_fonts(f);
}

/// `.semibold()` replaces egui's `.strong()`, which only changes the colour.
pub trait Semibold {
    fn semibold(self) -> Self;
}
impl Semibold for egui::RichText {
    fn semibold(self) -> Self {
        self.family(egui::FontFamily::Name(SEMIBOLD.into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use companion_core::access::contrast;

    fn rgb(c: Color32) -> [u8; 3] {
        [c.r(), c.g(), c.b()]
    }

    /// Every pair of text and background the interface really uses, in every look, reaches the
    /// WCAG AA level for normal text (4.5 : 1).
    #[test]
    fn every_look_is_readable() {
        for cb in [false, true] {
            for t in [Theme::Light, Theme::Dark, Theme::HighContrast] {
                let p = palette_for(t, cb);
                let pairs = [
                    ("text/bg", p.text, p.bg),
                    ("text/card", p.text, p.card),
                    ("text/panel", p.text, p.panel),
                    ("muted/bg", p.muted, p.bg),
                    ("muted/card", p.muted, p.card),
                    ("accent/bg", p.accent, p.bg),
                    ("on_accent/accent", p.on_accent, p.accent),
                    ("text/accent_soft", p.text, p.accent_soft),
                    ("text/bubble", p.text, p.bubble_user),
                    ("ok/bg", p.ok, p.bg),
                    ("ok/card", p.ok, p.card),
                    ("warn/warn_soft", p.warn, p.warn_soft),
                    ("warn/bg", p.warn, p.bg),
                    ("bad/bad_soft", p.bad, p.bad_soft),
                    ("bad/bg", p.bad, p.bg),
                ];
                for (name, a, b) in pairs {
                    let r = contrast(rgb(a), rgb(b));
                    assert!(r >= 4.5, "{t:?} colour_blind={cb} {name}: {r:.2}");
                }
            }
        }
    }

    #[test]
    fn colour_blind_palette_keeps_good_and_bad_apart_by_more_than_hue() {
        // Good and bad differ clearly in brightness too, so they stay apart in greyscale.
        for t in [Theme::Light, Theme::Dark] {
            let p = palette_for(t, true);
            let r = contrast(rgb(p.ok), rgb(p.bad));
            assert!(r >= 1.15, "{t:?}: {r:.2}");
            assert_ne!(p.ok, palette_for(t, false).ok);
        }
    }
}
