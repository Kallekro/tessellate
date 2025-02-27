use std::ops::Range;
use wgpu::util::DeviceExt;

use crate::model;

const VOXEL_WIDTH: f32 = 16.0;
const VOXEL_HEIGHT: f32 = 16.0;

struct Vertex {
    pub position: [f32; 3],
    pub tex_coords: [f32; 2],
    pub normal: [f32; 3],
}

const VERTICES: &[Vertex] = &[
    // Back face (z = -1)
    Vertex { position: [1.0, 1.0, -1.0], tex_coords: [1.0, 1.0], normal: [0.0, 0.0, -1.0] },    // top-right
    Vertex { position: [1.0, -1.0, -1.0], tex_coords: [1.0, 0.0], normal: [0.0, 0.0, -1.0] },   // bottom-right
    Vertex { position: [-1.0, -1.0, -1.0], tex_coords: [0.0, 0.0], normal: [0.0, 0.0, -1.0] },  // bottom-left
    Vertex { position: [-1.0, 1.0, -1.0], tex_coords: [0.0, 1.0], normal: [0.0, 0.0, -1.0] },   // top-left

    // Front face (z = 1)
    Vertex { position: [1.0, 1.0, 1.0], tex_coords: [1.0, 1.0], normal: [0.0, 0.0, 1.0] },     // top-right
    Vertex { position: [1.0, -1.0, 1.0], tex_coords: [1.0, 0.0], normal: [0.0, 0.0, 1.0] },    // bottom-right
    Vertex { position: [-1.0, -1.0, 1.0], tex_coords: [0.0, 0.0], normal: [0.0, 0.0, 1.0] },   // bottom-left
    Vertex { position: [-1.0, 1.0, 1.0], tex_coords: [0.0, 1.0], normal: [0.0, 0.0, 1.0] },    // top-left

    // Top face (y = 1)
    Vertex { position: [1.0, 1.0, 1.0], tex_coords: [1.0, 1.0], normal: [0.0, 1.0, 0.0] },     // front-right
    Vertex { position: [1.0, 1.0, -1.0], tex_coords: [1.0, 0.0], normal: [0.0, 1.0, 0.0] },    // back-right
    Vertex { position: [-1.0, 1.0, -1.0], tex_coords: [0.0, 0.0], normal: [0.0, 1.0, 0.0] },   // back-left
    Vertex { position: [-1.0, 1.0, 1.0], tex_coords: [0.0, 1.0], normal: [0.0, 1.0, 0.0] },    // front-left

    // Bottom face (y = -1)
    Vertex { position: [1.0, -1.0, 1.0], tex_coords: [1.0, 1.0], normal: [0.0, -1.0, 0.0] },    // front-right
    Vertex { position: [1.0, -1.0, -1.0], tex_coords: [1.0, 0.0], normal: [0.0, -1.0, 0.0] },   // back-right
    Vertex { position: [-1.0, -1.0, -1.0], tex_coords: [0.0, 0.0], normal: [0.0, -1.0, 0.0] },  // back-left
    Vertex { position: [-1.0, -1.0, 1.0], tex_coords: [0.0, 1.0], normal: [0.0, -1.0, 0.0] },   // front-left

    // Right face (x = 1)
    Vertex { position: [1.0, 1.0, 1.0], tex_coords: [1.0, 1.0], normal: [1.0, 0.0, 0.0] },     // front-top
    Vertex { position: [1.0, -1.0, 1.0], tex_coords: [1.0, 0.0], normal: [1.0, 0.0, 0.0] },    // front-bottom
    Vertex { position: [1.0, -1.0, -1.0], tex_coords: [0.0, 0.0], normal: [1.0, 0.0, 0.0] },   // back-bottom
    Vertex { position: [1.0, 1.0, -1.0], tex_coords: [0.0, 1.0], normal: [1.0, 0.0, 0.0] },    // back-top

    // Left face (x = -1)
    Vertex { position: [-1.0, 1.0, 1.0], tex_coords: [1.0, 1.0], normal: [-1.0, 0.0, 0.0] },    // front-top
    Vertex { position: [-1.0, -1.0, 1.0], tex_coords: [1.0, 0.0], normal: [-1.0, 0.0, 0.0] },   // front-bottom
    Vertex { position: [-1.0, -1.0, -1.0], tex_coords: [0.0, 0.0], normal: [-1.0, 0.0, 0.0] },  // back-bottom
    Vertex { position: [-1.0, 1.0, -1.0], tex_coords: [0.0, 1.0], normal: [-1.0, 0.0, 0.0] },   // back-top
];

const INDICES: &[u16] = &[
    0, 1, 2, 2, 3, 0,       // Back face
    4, 6, 5, 6, 4, 7,       // Front face
    8, 9, 10, 10, 11, 8,    // Top face
    12, 14, 13, 14, 12, 15, // Bottom face
    16, 17, 18, 18, 19, 16, // Right face
    20, 22, 21, 22, 20, 23, // Left face
];

pub struct Voxel {
    pub position: [f32; 3],
    // pub material: &'a model::Material,
}

impl Voxel {
    // pub fn new(position: [f32; 3], material: &'a model::Material) -> Self {
    //     Self { position, material }
    // }
}

pub struct VoxelModel {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub material: model::Material,
    pub bind_group: wgpu::BindGroup,
}

impl VoxelModel {
    pub fn new(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        material: model::Material,
    ) -> Self {
        let vertices = (0..VERTICES.len())
            .map(|i| model::ModelVertex {
                position: VERTICES[i].position,
                tex_coords: VERTICES[i].tex_coords,
                normal: VERTICES[i].normal,
            })
            .collect::<Vec<_>>();

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&material.diffuse_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&material.diffuse_texture.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&material.normal_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&material.normal_texture.sampler),
                },
            ],
        });

        Self {
            vertex_buffer,
            index_buffer,
            material,
            bind_group,
        }
    }
}

pub trait DrawVoxelModel<'a> {
    fn draw_voxel_model(
        &mut self,
        voxel_model: &'a VoxelModel,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    );

    fn draw_voxel_model_instanced(
        &mut self,
        voxel_model: &'a VoxelModel,
        instances: Range<u32>,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    );
}

impl<'a, 'b> DrawVoxelModel<'b> for wgpu::RenderPass<'a>
where
    'b: 'a,
{
    fn draw_voxel_model(
        &mut self,
        voxel_model: &'b VoxelModel,
        camera_bind_group: &'b wgpu::BindGroup,
        light_bind_group: &'b wgpu::BindGroup,
    ) {
        self.draw_voxel_model_instanced(voxel_model, 0..1, camera_bind_group, light_bind_group);
    }

    fn draw_voxel_model_instanced(
        &mut self,
        voxel_model: &'a VoxelModel,
        instances: Range<u32>,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    ) {
        self.set_vertex_buffer(0, voxel_model.vertex_buffer.slice(..));
        self.set_index_buffer(
            voxel_model.index_buffer.slice(..),
            wgpu::IndexFormat::Uint16,
        );
        self.set_bind_group(0, &voxel_model.bind_group, &[]);
        self.set_bind_group(1, camera_bind_group, &[]);
        self.set_bind_group(2, light_bind_group, &[]);
        self.draw_indexed(0..INDICES.len() as u32, 0, instances);
    }
}

// pub struct VoxelChunk<'a> {
//     voxels: [[Voxel<'a>; 32]; 32],
// }
