// Vertex shader
struct CameraUniform {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
};
@group(1) @binding(0)
var<uniform> camera: CameraUniform;

struct Light {
    position: vec3<f32>,
    color: vec3<f32>,
}
@group(2) @binding(0)
var<uniform> light: Light;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) tangent: vec3<f32>,
    @location(4) bitangent: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) world_position: vec3<f32>,
    @location(2) world_normal: vec3<f32>,    // Pass the world normal directly
    @location(3) world_tangent: vec3<f32>,   // Pass the world tangent directly
    @location(4) world_bitangent: vec3<f32>, // Pass the world bitangent directly
}

struct InstanceInput {
    @location(5) model_matrix_0: vec4<f32>,
    @location(6) model_matrix_1: vec4<f32>,
    @location(7) model_matrix_2: vec4<f32>,
    @location(8) model_matrix_3: vec4<f32>,
    @location(9) normal_matrix_0: vec3<f32>,
    @location(10) normal_matrix_1: vec3<f32>,
    @location(11) normal_matrix_2: vec3<f32>,
};

@vertex
fn vs_main(
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );
    let normal_matrix = mat3x3<f32>(
        instance.normal_matrix_0,
        instance.normal_matrix_1,
        instance.normal_matrix_2,
    );

    // Transform vertices to world space
    let world_position = (model_matrix * vec4<f32>(model.position, 1.0)).xyz;

    // Calculate world space vectors
    let world_normal = normalize(normal_matrix * model.normal);
    let world_tangent = normalize(normal_matrix * model.tangent);
    let world_bitangent = normalize(normal_matrix * model.bitangent);

    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(world_position, 1.0);
    out.tex_coords = model.tex_coords;
    out.world_position = world_position;
    out.world_normal = world_normal;
    out.world_tangent = world_tangent;
    out.world_bitangent = world_bitangent;

    return out;
}

// Fragment shader
@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;
@group(0)@binding(2)
var t_normal: texture_2d<f32>;
@group(0) @binding(3)
var s_normal: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let object_color: vec4<f32> = textureSample(t_diffuse, s_diffuse, in.tex_coords);
    let normal_map: vec4<f32> = textureSample(t_normal, s_normal, in.tex_coords);

    // Convert normal from [0,1] to [-1,1] range
    var normal_xyz = normal_map.xyz * 2.0 - 1.0;

    // IMPORTANT: Flip the Y component for correct orientation
    // This fixes the inverted normals issue
    // normal_xyz.y = -normal_xyz.y;

    // Orthogonalize the TBN vectors
    let N = normalize(in.world_normal);
    let T = normalize(in.world_tangent - N * dot(N, in.world_tangent));
    let B = normalize(cross(N, T));

    // Construct TBN matrix (tangent → world space transform)
    let TBN = mat3x3<f32>(T, B, N);

    // Transform normal from tangent space to world space
    let world_normal = normalize(TBN * normal_xyz);

    // Lighting calculations in world space
    let light_dir = normalize(light.position - in.world_position);
    let view_dir = normalize(camera.view_pos.xyz - in.world_position);
    let half_dir = normalize(view_dir + light_dir);

    // Ambient lighting
    let ambient_strength = 0.1;
    let ambient_color = light.color * ambient_strength;

    // Diffuse lighting
    let diffuse_strength = max(dot(world_normal, light_dir), 0.0);
    let diffuse_color = light.color * diffuse_strength;

    // Specular lighting
    let specular_exponent = 32.0;
    let specular_intensity = 0.3;
    let specular_strength = specular_intensity * pow(max(dot(world_normal, half_dir), 0.0), specular_exponent);
    let specular_color = specular_strength * light.color;

    let result = (ambient_color + diffuse_color + specular_color) * object_color.xyz;
    return vec4<f32>(result, object_color.a);
}