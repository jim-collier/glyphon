use crate::{Cache, Params, Resolution};
use std::{mem, slice};
use wgpu::{BindGroup, Buffer, BufferDescriptor, BufferUsages, Device, Queue};

/// Controls the visible area of all text for a given renderer. Any text outside of the visible
/// area will be clipped.
///
/// Many projects will only ever need a single `Viewport`, but it is possible to create multiple
/// `Viewport`s if you want to render text to specific areas within a window (without having to)
/// bound each `TextArea`).
#[derive(Debug)]
pub struct Viewport {
    params: Params,
    params_buffer: Buffer,
    pub(crate) bind_group: BindGroup,
}

impl Viewport {
    /// Creates a new `Viewport` with the given `device` and `cache`.
    pub fn new(device: &Device, cache: &Cache) -> Self {
        let params = Params {
            screen_resolution: Resolution {
                width: 0,
                height: 0,
            },
            text_fg: 0.0,
            text_bg: 0.0,
            text_blend: 0.0,
            _pad: [0; 3],
        };

        let params_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("glyphon params"),
            size: mem::size_of::<Params>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = cache.create_uniforms_bind_group(device, &params_buffer);

        Self {
            params,
            params_buffer,
            bind_group,
        }
    }

    /// Updates the `Viewport` with the given `resolution`.
    pub fn update(&mut self, queue: &Queue, resolution: Resolution) {
        if self.params.screen_resolution != resolution {
            self.params.screen_resolution = resolution;
            self.write(queue);
        }
    }

    /// Bends glyph coverage so the finished pixel lands where an sRGB blend of
    /// `fg` over `bg` would have put it.
    ///
    /// On a linear target, coverage blends in linear light: a half covered
    /// pixel comes out near three quarters brightness whichever way round the
    /// two colors are. That is a strong edge on a dark background and hardly
    /// any ink on a light one, so dark text reads a weight lighter than the
    /// font was drawn for. Almost every other program blends text in sRGB, and
    /// this puts the result back there without a second encode: the output is
    /// still linear and the target is encoded once, at the surface.
    ///
    /// `fg` and `bg` are sRGB grays of the same brightness as the real colors,
    /// since one alpha has to serve all three channels. `blend` is how much of
    /// the correction to apply; 0 leaves coverage exactly as it was, which is
    /// also what happens when `fg` is not darker than `bg`.
    pub fn set_text_blend(&mut self, queue: &Queue, fg: f32, bg: f32, blend: f32) {
        if (self.params.text_fg, self.params.text_bg, self.params.text_blend) != (fg, bg, blend) {
            self.params.text_fg = fg;
            self.params.text_bg = bg;
            self.params.text_blend = blend;
            self.write(queue);
        }
    }

    /// Returns the text pair and blend amount coverage is corrected with.
    pub fn text_blend(&self) -> (f32, f32, f32) {
        (
            self.params.text_fg,
            self.params.text_bg,
            self.params.text_blend,
        )
    }

    fn write(&self, queue: &Queue) {
        queue.write_buffer(&self.params_buffer, 0, unsafe {
            slice::from_raw_parts(
                &self.params as *const Params as *const u8,
                mem::size_of::<Params>(),
            )
        });
    }

    /// Returns the current resolution of the `Viewport`.
    pub fn resolution(&self) -> Resolution {
        self.params.screen_resolution
    }
}
