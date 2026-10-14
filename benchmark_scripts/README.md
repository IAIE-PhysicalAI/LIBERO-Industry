# Benchmark scripts

## BDDL Generation

Apply these invariants when generating or modifying LIBERO-Industry BDDL files.

1. Count repeated object instances explicitly. Do not use a dictionary
   comprehension that overwrites duplicate keys:
   `obj_counts[obj] = obj_counts.get(obj, 0) + 1`.
2. Track and increment region suffixes such as `_1` and `_2` during
   `init_states` allocation. Do not hardcode `_1` inside a loop.
3. Use the canonical predicate forms:
   - Conveyor: `("AtSpeed", "conveyor_belt_1", str(speed), "steady", f"takt_{takt}", f"offset_{offset}")`
   - Ghost: `("LoadSlot", "conveyor_ghost_1", inlay, axis, "pulse", str(speed), f"service_{service}")`
   - Moving target goal: `Routed`
   - Static plate goal: `AlignedOn`

Stage generated artifacts outside the live benchmark directories. Promote BDDL
and initial-state artifacts together only after representative preview and reset
validation.

### Public benchmark track

The unsuffixed suite names are reserved for the single-arm-compatible benchmark:

- `libero_industry_s2d`
- `libero_industry_d2s`
- `libero_industry_d2d`
