use std::rc::Rc;

use criterion::{Criterion, async_executor::FuturesExecutor, criterion_group, criterion_main};
use wgpu::{TextureDescriptor, TextureFormat, TextureUsages, util::DeviceExt};

fn benchmark_encode_ansi(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();
    let (texture, ansi_encoder) = runtime.block_on(async {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let device = Rc::new(device);
        let queue = Rc::new(queue);

        const SIZE: (usize, usize) = (1920, 1080);
        let pixels: Vec<u8> = gpu_ansi_encoder::gen_pixels(SIZE.0, SIZE.1);
        let texture = device.create_texture_with_data(
            &queue,
            &TextureDescriptor {
                label: Some("test_red"),
                size: wgpu::Extent3d {
                    width: SIZE.0.try_into().unwrap(),
                    height: SIZE.1.try_into().unwrap(),
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

        let ansi_encoder = gpu_ansi_encoder::GpuAnsiEncoder::new(device, queue)
            .await
            .unwrap();
        (texture, ansi_encoder)
    });

    c.bench_function("encode_ansi", |b| {
        b.to_async(FuturesExecutor)
            .iter(|| ansi_encoder.ansi_from_texture(&texture))
    });
}

criterion_group!(encode_ansi_benchmarks, benchmark_encode_ansi);
criterion_main!(encode_ansi_benchmarks);
