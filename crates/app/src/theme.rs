//! Colours, spacing and sizes. Three looks: light, dark and a high-contrast one
//! for low vision. Targets are large and spaced out on purpose.

use companion_core::Theme;
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

pub fn palette(theme: Theme) -> Palette {
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

pub fn apply(ctx: &egui::Context, theme: Theme, zoom: f32) {
    let p = palette(theme);
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
    ctx.global_style_mut(|s| {
        s.spacing.item_spacing = vec2(10.0, 9.0);
        s.spacing.button_padding = vec2(14.0, 8.0);
        s.spacing.interact_size.y = 34.0;
        s.spacing.text_edit_width = 320.0;
        s.spacing.scroll.floating = true;
        s.interaction.tooltip_delay = 0.4;
    });
    ctx.set_zoom_factor(zoom.clamp(0.8, 2.0));
}

/// Name of the font family used for emphasis (egui has no synthetic bold).
pub const SEMIBOLD: &str = "semibold";

/// Inter for text and JetBrains Mono for code, both under the SIL OFL 1.1 (see licenses/).
/// The built-in fonts stay as fallbacks for symbols and other scripts.
pub fn install_fonts(ctx: &egui::Context) {
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
        .insert(0, "inter".into());
    f.families
        .get_mut(&FontFamily::Monospace)
        .unwrap()
        .insert(0, "jbmono".into());
    let mut bold = vec!["inter-semibold".to_owned()];
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
