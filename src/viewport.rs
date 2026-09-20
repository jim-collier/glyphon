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
            coverage_gamma: 1.0,
            _pad: 0,
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

    /// Sets the exponent applied to glyph coverage before it becomes alpha.
    ///
    /// The default, 1.0, leaves coverage as the rasterizer produced it. A value
    /// below 1.0 raises partly covered pixels, which is what dark text on a
    /// light background needs when the two are blended in linear light: without
    /// it a half covered pixel carries far less ink than the eye expects, and
    /// the text reads thin. A value above 1.0 does the reverse.
    pub fn set_coverage_gamma(&mut self, queue: &Queue, gamma: f32) {
        if self.params.coverage_gamma != gamma {
            self.params.coverage_gamma = gamma;
            self.write(queue);
        }
    }

    /// Returns the exponent applied to glyph coverage.
    pub fn coverage_gamma(&self) -> f32 {
        self.params.coverage_gamma
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
