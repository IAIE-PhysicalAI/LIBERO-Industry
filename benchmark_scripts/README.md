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

### Experimental bimanual BDDL generation

This branch can stage the harder bimanual BDDL definitions, but it does not yet
provide registered bimanual benchmark selectors, a dual-arm action/observation
interface, or dual-arm initial states. It is therefore a development path, not
a runnable dual-arm benchmark release.

Stage the BDDLs outside the live benchmark directories:

```powershell
python notebooks/industry/autogen/regenerate_industry_benchmark.py build `
  --run-id bimanual-example `
  --track bimanual
```

Preview one representative staged task before expanding validation:

```powershell
python notebooks/industry/autogen/regenerate_industry_benchmark.py preview `
  --run-id bimanual-example `
  --suite S2D `
  --level 2 `
  --per-level 1
```

To turn these definitions into a runnable dual-arm benchmark:

1. Implement and register the exact dual-arm robot, grippers, controller, and
   action/observation contract.
2. Register the bimanual suite names and their BDDL task maps.
3. Add an embodiment-specific initial-state profile and generate target-native
   states; do not reuse flattened single-arm states.
4. Validate BDDL parsing, one deterministic reset, and one video-enabled task
   before generating or validating the full benchmark.
5. Document the policy input/output interface and evaluation contract for that
   embodiment.

The intended generation-only suite names are
`libero_industry_{s2d,d2s,d2d}_bimanual`. The bimanual track preserves the
harder Level-2 pose and V-groove variants and the Level-3 inverted D2D object
distribution while keeping the temporal schedule paired with the public track.
