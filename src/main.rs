use anyhow::Result;
use std::rc::Rc;
use wgpu::{TextureFormat, TextureUsages, util::DeviceExt, wgt::TextureDescriptor};

mod gpu_ansi_encoder;

#[tokio::main]
async fn main() -> Result<()> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await?;
    let device = Rc::new(device);
    let queue = Rc::new(queue);

    let size = term_size::dimensions().ok_or_else(|| anyhow::anyhow!("failed to get term size"))?;
    let size = (size.0, size.1 * 2);
    let size = (20, 20);

    let pixels: Vec<u8> = gpu_ansi_encoder::gen_pixels(size.0, size.1);
    let texture = device.create_texture_with_data(
        &queue,
        &TextureDescriptor {
            label: Some("test_red"),
            size: wgpu::Extent3d {
                width: size.0.try_into().unwrap(),
                height: size.1.try_into().unwrap(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::Rgba8Uint,
            usage: TextureUsages::TEXTURE_BINDING,
            view_formats: &[TextureFormat::Rgba8Uint],
        },
        wgpu::wgt::TextureDataOrder::LayerMajor,
        &pixels,
    );

    let ansi_encoder = gpu_ansi_encoder::GpuAnsiEncoder::new(device, queue).await?;
    let s = ansi_encoder.ansi_from_texture(&texture).await?;
    print!("{s}");

    Ok(())
}
