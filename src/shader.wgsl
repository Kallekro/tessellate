// Vertex shader

struct Camera {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
}
@group(1) @binding(0)
var<uniform> camera: Camera;

struct Light {
    position: vec4<f32>,
    color: vec3<f32>,
    view_proj: mat4x4<f32>,
}
@group(2) @binding(0)
var<uniform> light: Light;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) normal: vec3<f32>,
}
struct InstanceInput {
    @location(5) model_matrix_0: vec4<f32>,
    @location(6) model_matrix_1: vec4<f32>,
    @location(7) model_matrix_2: vec4<f32>,
    @location(8) model_matrix_3: vec4<f32>,
    @location(9) normal_matrix_0: vec3<f32>,
    @location(10) normal_matrix_1: vec3<f32>,
    @location(11) normal_matrix_2: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) world_position: vec3<f32>,
}

struct ShadowVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
}

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
    var out: VertexOutput;
    out.tex_coords = model.tex_coords;
    out.world_normal = normal_matrix * model.normal;
    var world_position: vec4<f32> = model_matrix * vec4<f32>(model.position, 1.0);
    out.world_position = world_position.xyz;
    out.clip_position = camera.view_proj * world_position;
    return out;
}

// Add shadow vertex shader entry point
@vertex
fn shadow_vs_main(
    model: VertexInput,
    instance: InstanceInput,
) -> ShadowVertexOutput {
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );

    var output: ShadowVertexOutput;
    output.clip_position = light.view_proj * model_matrix * vec4<f32>(model.position, 1.0);
    return output;
}

// Fragment shader

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;
@group(3) @binding(0)
var t_shadow: texture_depth_2d;
@group(3) @binding(1)
var s_shadow: sampler_comparison;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let object_color: vec4<f32> = textureSample(t_diffuse, s_diffuse, in.tex_coords);

    // Transform fragment position into light space
    // let light_space_pos = light.view_proj * vec4<f32>(in.world_position, 1.0);

    // // Perspective divide and convert to UV coordinates
    // let ndc = light_space_pos.xyz / light_space_pos.w;
    // let proj_coords = vec3<f32>(
    //     ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5),
    //     ndc.z
    // );

    // // Sample shadow map
    // let shadow_depth = textureSampleCompare(t_shadow, s_shadow, proj_coords.xy, proj_coords.z);

    // // Compare depths with bias to avoid shadow acne
    // let bias = 0.0005;
    // let shadow = select(1.0, 0.5, proj_coords.z - bias > shadow_depth);
    // let shadow = 1.0;
// Transform world position to light space with proper perspective divide
    let light_dir = normalize(light.position.xyz - in.world_position);
    let view_dir = normalize(camera.view_pos.xyz - in.world_position);
    let half_dir = normalize(view_dir + light_dir);

    let light_pos = light.view_proj * vec4<f32>(in.world_position, 1.0);
    let ndc = light_pos.xyz / light_pos.w;

    var shadow = 1.0;
    // Check if fragment is in light's view
    if (abs(ndc.x) <= 1.0 && abs(ndc.y) <= 1.0 && ndc.z <= 1.0) {
        let shadow_coords = vec2<f32>(ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5));

        var visibility = 0.0;
        let bias = max(0.001 * (1.0 - dot(in.world_normal, light_dir)), 0.0001);

        let texel_size = 1.0 / f32(2048);
        for(var y = -1; y <= 1; y += 1) {
            for(var x = -1; x <= 1; x += 1) {
                let offset = vec2<f32>(f32(x), f32(y)) * texel_size;
                visibility += textureSampleCompare(
                    t_shadow,
                    s_shadow,
                    shadow_coords + offset,
                    ndc.z - bias
                );
            }
        }
        visibility /= 9.0;

        // Fade out shadows at texture edges
        let edge = 0.1;
        let fade = smoothstep(1.0 - edge, 1.0, max(abs(ndc.x), abs(ndc.y)));
        visibility = mix(visibility, 1.0, fade);

        shadow = visibility;
    }

    // We don't need (or want) much ambient light, so 0.1 is fine
    let ambient_strength = 0.1;
    let ambient_color = light.color * ambient_strength;

    let diffuse_strength = max(dot(in.world_normal, light_dir), 0.0);
    let diffuse_color = light.color * diffuse_strength * shadow;

    let specular_strength = pow(max(dot(in.world_normal, half_dir), 0.0), 32.0);
    let specular_color = specular_strength * light.color * shadow;

    let result = (ambient_color + diffuse_color + specular_color) * object_color.xyz;

    return vec4<f32>(result, object_color.a);
}