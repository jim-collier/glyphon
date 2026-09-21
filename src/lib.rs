//! Glyphon provides a simple way to render 2D text with [wgpu], [cosmic-text] and [etagere].
//!
//! [wpgu]: https://github.com/gfx-rs/wgpu
//! [cosmic-text]: https://github.com/pop-os/cosmic-text
//! [etagere]: https://github.com/nical/etagere

mod cache;
mod custom_glyph;
mod error;
mod text_atlas;
mod text_render;
mod viewport;

pub use cache::Cache;
pub use custom_glyph::{
    ContentType, CustomGlyph, CustomGlyphId, RasterizeCustomGlyphRequest, RasterizedCustomGlyph,
};
pub use error::{PrepareError, RenderError};
pub use text_atlas::{ColorMode, TextAtlas};
pub use text_render::TextRenderer;
pub use viewport::Viewport;

// Re-export all top-level types from `cosmic-text` for convenience.
#[doc(no_inline)]
pub use cosmic_text::{
    self, fontdb, Action, Affinity, Attrs, AttrsList, AttrsOwned, Buffer, BufferLine, CacheKey,
    Color, Command, Cursor, Edit, Editor, Family, FamilyOwned, Font, FontSystem, LayoutCursor,
    LayoutGlyph, LayoutLine, LayoutRun, LayoutRunIter, Metrics, ShapeGlyph, ShapeLine, ShapeSpan,
    ShapeWord, Shaping, Stretch, Style, SubpixelBin, SwashCache, SwashContent, SwashImage, Weight,
    Wrap,
};

use etagere::AllocId;
use wgpu::{Device, Queue};

pub(crate) enum GpuCacheStatus {
    InAtlas {
        x: u16,
        y: u16,
        content_type: ContentType,
    },
    SkipRasterization,
}

pub(crate) struct GlyphDetails {
    width: u16,
    height: u16,
    gpu_cache: GpuCacheStatus,
    atlas_id: Option<AllocId>,
    top: i16,
    left: i16,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct GlyphToRender {
    pos: [i32; 2],
    dim: [u16; 2],
    uv: [u16; 2],
    color: u32,
    content_type_with_srgb: [u16; 2],
    depth: f32,
}

/// The screen resolution to use when rendering text.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Resolution {
    /// The width of the screen in pixels.
    pub width: u32,
    /// The height of the screen in pixels.
    pub height: u32,
}

#[repr(C)]
// No Eq: the three text fields are floats. Nothing compares the whole struct.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Params {
    screen_resolution: Resolution,
    /// The text and its background as sRGB grays of the same brightness, and
    /// how much of the coverage correction to apply. 0 leaves coverage as the
    /// rasterizer produced it. See `Viewport::set_text_blend`.
    text_fg: f32,
    text_bg: f32,
    text_blend: f32,
    _pad: [u32; 3],
}

/// Controls the visible area of the text. Any text outside of the visible area will be clipped.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TextBounds {
    /// The position of the left edge of the visible area.
    pub left: i32,
    /// The position of the top edge of the visible area.
    pub top: i32,
    /// The position of the right edge of the visible area.
    pub right: i32,
    /// The position of the bottom edge of the visible area.
    pub bottom: i32,
}

/// The default visible area doesn't clip any text.
impl Default for TextBounds {
    fn default() -> Self {
        Self {
            left: i32::MIN,
            top: i32::MIN,
            right: i32::MAX,
            bottom: i32::MAX,
        }
    }
}

/// A text area containing text to be rendered along with its overflow behavior.
#[derive(Clone)]
pub struct TextArea<'a> {
    /// The buffer containing the text to be rendered.
    pub buffer: &'a Buffer,
    /// The left edge of the buffer.
    pub left: f32,
    /// The top edge of the buffer.
    pub top: f32,
    /// The scaling to apply to the buffer.
    pub scale: f32,
    /// The visible bounds of the text area. This is used to clip the text and doesn't have to
    /// match the `left` and `top` values.
    pub bounds: TextBounds,
    /// The default color of the text area.
    pub default_color: Color,
    /// Additional custom glyphs to render.
    pub custom_glyphs: &'a [CustomGlyph],
}

pub(crate) struct State<'a> {
    pub(crate) device: &'a Device,
    pub(crate) queue: &'a Queue,
}

#[cfg(test)]
mod tests {
    use super::Params;

    fn srgb_to_linear(c: f32) -> f32 {
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    }

    // The mask arm of the fragment stage, in Rust. Nothing here runs WGSL, so
    // the curve is checked through this and the shader's own text is held
    // against it below.
    fn corrected(coverage: f32, fg: f32, bg: f32, blend: f32) -> f32 {
        if blend == 0.0 || fg >= bg {
            return coverage;
        }
        let (fg_l, bg_l) = (srgb_to_linear(fg), srgb_to_linear(bg));
        let blended = coverage * fg + (1.0 - coverage) * bg;
        let matched = ((srgb_to_linear(blended) - bg_l) / (fg_l - bg_l)).clamp(0.0, 1.0);
        coverage + (matched - coverage) * blend
    }

    // The uniform reaches the GPU as raw bytes, so the layout is pinned. It was
    // 16 with the one exponent field in the old padding; the pair plus the
    // amount take a second 16.
    //   assert_eq!(std::mem::size_of::<Params>(), 16);
    //   assert_eq!(std::mem::offset_of!(Params, coverage_gamma), 8);
    #[test]
    fn the_text_fields_follow_the_resolution() {
        assert_eq!(std::mem::size_of::<Params>(), 32);
        assert_eq!(std::mem::offset_of!(Params, text_fg), 8);
        assert_eq!(std::mem::offset_of!(Params, text_bg), 12);
        assert_eq!(std::mem::offset_of!(Params, text_blend), 16);
    }

    // The whole point: at full blend the composite is what an sRGB blend of the
    // pair would have been, which is the weight the font was drawn for.
    #[test]
    fn a_full_blend_matches_an_srgb_blend() {
        let (fg, bg) = (0.196, 0.960); // SilkTerm's light theme, as grays
        let (fg_l, bg_l) = (srgb_to_linear(fg), srgb_to_linear(bg));
        for step in 0u8..=20 {
            let coverage = f32::from(step) / 20.0;
            let a = corrected(coverage, fg, bg, 1.0);
            let out = a * fg_l + (1.0 - a) * bg_l;
            let want = srgb_to_linear(coverage * fg + (1.0 - coverage) * bg);
            assert!(
                (out - want).abs() < 1e-5,
                "coverage {coverage}: {out} against {want}"
            );
        }
    }

    #[test]
    fn nothing_is_corrected_without_a_blend_or_against_lighter_text() {
        for coverage in [0.0, 0.25, 0.5, 0.75, 1.0] {
            assert_eq!(corrected(coverage, 0.2, 0.9, 0.0), coverage);
            assert_eq!(corrected(coverage, 0.9, 0.2, 1.0), coverage);
            assert_eq!(corrected(coverage, 0.5, 0.5, 1.0), coverage);
        }
    }

    // A partial blend sits between the two, and the ends are exact.
    #[test]
    fn a_partial_blend_sits_between_the_two() {
        let (fg, bg) = (0.0, 1.0);
        let half = corrected(0.5, fg, bg, 0.5);
        assert!(half > 0.5 && half < corrected(0.5, fg, bg, 1.0));
        assert_eq!(corrected(0.5, fg, bg, 0.0), 0.5);
    }

    // Nothing here runs WGSL, so the shader's own text is what holds the mirror
    // above honest.
    #[test]
    fn the_shader_matches_the_mirror() {
        let src = include_str!("shader.wgsl");
        for want in [
            "text_fg: f32,",
            "text_bg: f32,",
            "text_blend: f32,",
            "params.text_blend != 0.0 && params.text_fg < params.text_bg",
            "coverage * params.text_fg + (1.0 - coverage) * params.text_bg",
            "clamp((srgb_to_linear(blended) - bg_l) / (fg_l - bg_l), 0.0, 1.0)",
            "mix(coverage, matched, params.text_blend)",
        ] {
            assert!(src.contains(want), "shader is missing `{want}`");
        }
    }
}
