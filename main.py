from scene import Scene
import taichi as ti
from taichi.math import *
scene = Scene(voxel_edges=.1, exposure=1.2)
scene.set_floor(-1, (1, 1, 1))
scene.set_directional_light((0.8,1,-0.5), .1, (1, 1, 1.05))
scene.set_background_color((1, 1, 1))
n = 64

@ti.func
def rot(t):
    c = ti.cos(t); s = ti.sin(t)
    return ti.math.mat2([[c, s], [-s, c]])

@ti.func
def dbar(p, a, b, r):
    pa = p - a; ba = b - a
    h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0)
    return distance(pa - ba * h, vec3(0)) - r

@ti.func
def tree(z):
    col = 0.
    d = ti.max(ti.abs(z.y + 1.) - 1., distance(z.xz, vec2(0)) - .13)
    for i in range(0, 8):
        scale = (i + 2) * 2.
        z.xz = rot(ti.floor(.5 + ti.atan2(z.x, -z.z) * scale / 6.28) * 6.28 / scale) @ z.xz
        z.yz = rot(-.5) @ z.yz
        bar = dbar(z.xyz + vec3(0., 0., 1.), vec3(0., 0., 1.), vec3(0., 2., 0.), .1)
        col += .1 * bar / scale
        d = ti.min(d, bar / scale)
        z.yz += vec2(-2, 1.)
        z *= 2.; scale *= 2.
    return (d, col)

@ti.func
def fold(p, a):
    p.x = ti.abs(p.x)
    k = vec2(ti.cos(a), ti.sin(a))
    for i in range(0, 5):
        p -= 2. * ti.min(0., ti.math.dot(p, k)) * k
        k = normalize(k - vec2(1, 0))
    return p

@ti.func
def apo(p):
    scale = 1.
    orb = vec4(1, 1, 1, 1) * 1000.
    for i in range(0, 8):
        p = -1. + 2. * ti.math.fract(0.5 * p + 0.5)
        r2 = ti.math.dot(p, p)
        if i >= 1:
            orb.x = ti.min(orb.x, ti.abs(p.x)); orb.y = ti.min(orb.y, ti.abs(p.y))
            orb.z = ti.min(orb.z, ti.abs(p.z));  orb.w = ti.min(orb.w, r2)
        k = 1.2 / r2
        p *= k; scale *= k
    return (.25 * ti.abs(p.y) / scale, orb)

@ti.func
def draw():
    for i, j, k in ti.ndrange((-n, n), (-n, 20), (-n, n)):
        if j > 0 and distance(vec2(i, k), vec2(0)) > 26:
            continue
        p = vec3(i, j + 109, k)
        p.xz = fold(p.xz, 0.05)
        p.yz = fold(p.yz, 0.225)
        res = apo(p * .032)
        color = mix(vec3(1., .5, .0), vec3(1), smoothstep(-n//2-20, -n//2, j ))
        if res[0] * 5. < .02:
            scene.set_voxel(ivec3(i, j, k), 1,
                            mix(color, vec3(.5, .1, .0), res[1].w * res[1].w))

    for i, j, k in ti.ndrange((-n, n), (-n, n), (-n, n)):
        res = tree(vec3(i, j+n+20, k) * .06)
        if res[0] < .02:
            scene.set_voxel(vec3(i, j, k), 1,
                            mix(vec3(.2, 1., .0) * ti.random(), vec3(.15,.1,.0),  res[1]))
        if j >= 20:
            scene.set_voxel(vec3(i, j, k), 0, vec3(0))

    for i, j, k in ti.ndrange((-n, n), (-n, n), (-n, n)):
        res = tree(vec3(i, j, k) * .1)
        if res[0] < .02:
            scene.set_voxel(vec3(i, j - n - 30, k), 1,
                            mix(vec3(.2, 1., .0) * ti.random(), vec3(.15, .1, .0), res[1]))

@ti.kernel
def initialize_voxels():
    draw()
initialize_voxels()
scene.finish()
