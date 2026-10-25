"""Optional CP-SAT placement; preserves fixed group sizes and internal transforms."""
import copy,math,time,threading
from .quality import units

def _solve_legacy(source,design,A,locked=(),budget_seconds=.4):
    try:from ortools.sat.python import cp_model
    except ImportError:return None,{'status':'unavailable'}
    m=cp_model.CpModel();cw,ch=design['canvas']['width'],design['canvas']['height'];groups=units(A,design);old={b.id:r for b,r in units(A,source)};vs={};cost=[]
    for b,r in groups:
        w,h=math.ceil(r.w),math.ceil(r.h)
        if w>cw or h>ch:return None,{'status':'oversized'}
        x=m.new_int_var(0,cw-w,'x'+b.id);y=m.new_int_var(0,ch-h,'y'+b.id);vs[b.id]=(x,y)
        if set(locked).intersection(b.ids):m.add(x==round(r.x));m.add(y==round(r.y))
        for v,ref,limit in ((x,r.x,cw),(y,r.y,ch)):
            d=m.new_int_var(0,2*limit,'d'+b.id+str(ref));m.add_abs_equality(d,v-round(ref));cost.append(d*max(1,round(b.importance*10)))
        m.add_hint(x,max(0,min(cw-w,round(r.x))));m.add_hint(y,max(0,min(ch-h,round(r.y))))
    for i,(a,r) in enumerate(groups):
        for b,s in groups[i+1:]:
            # Preserve intentional source overlaps, constrain only unrelated blocks.
            if old[a.id].intersection_area(old[b.id])>0:continue
            ax,ay=vs[a.id];bx,by=vs[b.id];bs=[m.new_bool_var(f'{i}-{b.id}-{k}') for k in range(4)]
            for cond,flag in zip((ax+math.ceil(r.w)+1<=bx,bx+math.ceil(s.w)+1<=ax,ay+math.ceil(r.h)+1<=by,by+math.ceil(s.h)+1<=ay),bs):m.add(cond).only_enforce_if(flag)
            m.add_bool_or(bs)
    m.minimize(sum(cost));solver=cp_model.CpSolver();solver.parameters.max_time_in_seconds=max(0,float(budget_seconds));solver.parameters.num_search_workers=1;solver.parameters.random_seed=7
    status=solver.solve(m)
    if status not in (cp_model.FEASIBLE,cp_model.OPTIMAL):return None,{'status':solver.status_name(status)}
    out=copy.deepcopy(design);by={e['id']:e for e in out['elements']}
    for b,r in groups:
        x,y=vs[b.id];dx,dy=solver.value(x)-r.x,solver.value(y)-r.y
        for eid in b.ids:by[eid]['x']+=dx;by[eid]['y']+=dy
    return out,{'status':solver.status_name(status),'objective':solver.objective_value}


def solve(source, design, A, locked=(), *, mode="legacy", budget_seconds=.4, cancelled=None):
    """Compatibility adapter. Joint sizing is deliberately opt-in.

    Locks freeze complete current-design groups containing a locked ID. They
    never silently shrink or move the locked geometry.
    """
    if mode not in ("legacy", "joint"):
        raise ValueError("solver mode must be 'legacy' or 'joint'")
    if mode == "joint":
        return solve_joint(source, design, A, locked, budget_seconds=budget_seconds,
                           cancelled=cancelled)
    if cancelled is not None and cancelled():
        return None, {"status": "CANCELLED"}
    return _solve_legacy(source, design, A, locked, budget_seconds)


def solve_joint(source, design, A, locked=(), *, budget_seconds=.4, cancelled=None):
    """Joint finite text sizing and selective non-overlap placement.

    Rigid multi-element blocks and rotations retain internal geometry; only
    independent unrotated text obtains measured configurations. A 0.01px grid
    and conservative extents protect export rounding. Source text/text overlap
    is never automatically treated as an intentional overlay.

    Return ``(None, event)`` on failure; the caller retains its verified
    incumbent. UNKNOWN never means INFEASIBLE. This solver certifies geometry
    and measured text fit, not visual quality or resource-complete rendering.
    """
    from .blocks import FLOORS
    from .canonical import rotated_extent
    from .configurations import enumerate_text_configurations,enumerate_group_configurations
    from .typography import measure

    start = time.monotonic()
    budget = float(budget_seconds)
    if not math.isfinite(budget) or budget < 0:
        raise ValueError("budget_seconds must be finite and nonnegative")
    event = {"mode": "joint", "grid": .01, "priority": "readability_then_movement",
             "geometry_verified": False, "configurations": {}, "conflicting_locks": []}

    def finish(out, status, **extra):
        return out, {**event, **extra, "status": status,
                     "elapsed_seconds": time.monotonic()-start}

    if cancelled is not None and cancelled():
        return finish(None, "CANCELLED")
    if budget == 0:
        return finish(None, "UNKNOWN", reason="budget exhausted before construction")
    try:
        from ortools.sat.python import cp_model
    except ImportError:
        return finish(None, "unavailable")
    cw, ch = float(design["canvas"]["width"]), float(design["canvas"]["height"])
    if not all(math.isfinite(v) and 0 < v <= 1e6 for v in (cw, ch)):
        return finish(None, "MODEL_INVALID", reason="unsupported canvas bounds")
    grid = 100
    W, H = math.floor(cw*grid), math.floor(ch*grid)
    groups = units(A, design)
    source_groups = {b.id: r for b, r in units(A, source)}
    original = {e["id"]: e for e in design["elements"]}
    if len(original) != len(design["elements"]):
        return finish(None, "MODEL_INVALID", reason="duplicate element IDs")
    used = [eid for b, _ in groups for eid in b.ids if eid in original]
    if len(used) != len(set(used)):
        return finish(None, "MODEL_INVALID", reason="overlapping group membership")
    locks = set(locked)
    unknown_locks = locks - original.keys()
    if unknown_locks:
        return finish(None, "MODEL_INVALID", reason="unknown locked IDs", conflicting_locks=sorted(unknown_locks))
    from collections.abc import Mapping
    pins=({eid:dict(rect) for eid,rect in locked.items()} if isinstance(locked,Mapping) else
          {eid:{key:original[eid][key] for key in ('x','y','width','height','font_size') if key in original[eid]} for eid in locks})
    if any(not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value)
           for rectangle in pins.values() for value in rectangle.values()):
        return finish(None,'MODEL_INVALID',reason='non-finite child lock')
    policies={p.group_id:p for p in getattr(A,'policies',()) if p.parent is None}
    m = cp_model.CpModel()
    variables = {}
    choices = {}
    group_choices=set()
    quality_cost, movement_cost = [], []
    frozen = set()
    for index, (b, r) in enumerate(groups):
        members = [original[eid] for eid in b.ids if eid in original]
        is_locked = bool(locks.intersection(b.ids))
        configs = ()
        policy=policies.get(b.id)
        if policy is not None and set(policy.member_ids)==set(b.ids) and len(members)>1:
            configs=enumerate_group_configurations(policy,source,A,{'width':cw,'height':ch},locks=pins)
            if not configs:
                return finish(None,'INFEASIBLE',reason='no measured group configuration satisfies policies, resources and locks',
                              ids=b.ids,conflicting_locks=sorted(locks.intersection(b.ids)),policy_id=policy.group_id)
            group_choices.add(b.id)
            choices[b.id]=configs
            widths=[math.ceil((c.anchor[0]+c.width)*grid)-math.floor(c.anchor[0]*grid) if c.anchor else math.ceil(c.width*grid)+2 for c in configs]
            heights=[math.ceil((c.anchor[1]+c.height)*grid)-math.floor(c.anchor[1]*grid) if c.anchor else math.ceil(c.height*grid)+2 for c in configs]
            select=m.new_int_var(0,len(configs)-1,f'configuration_{index}')
            width=m.new_int_var(min(widths),max(widths),f'width_{index}')
            height=m.new_int_var(min(heights),max(heights),f'height_{index}')
            m.add_element(select,widths,width);m.add_element(select,heights,height)
            penalty=[c.readability_penalty for c in configs]
            readability=m.new_int_var(min(penalty),max(penalty),f'readability_{index}')
            m.add_element(select,penalty,readability)
            quality_cost.append(readability*max(1,round(b.importance*10)))
            event['configurations'][b.id]=len(configs)
        elif len(members) == 1 and members[0]["type"] == "text" and not members[0].get("rotation") and not is_locked:
            e = members[0]
            floor = min(int(e.get("font_size", 12)), FLOORS.get(b.role, 8))
            configs = enumerate_text_configurations(e, max_width=cw, max_height=ch,
                fs_min=max(1, floor), fs_max=max(1, int(e.get("font_size", 12))),
                allow_estimated=False)
            if not configs:
                return finish(None, "UNVERIFIED", reason="no exact-font readable text configuration", ids=b.ids)
            choices[b.id] = configs
            widths, heights = [c.width*grid for c in configs], [c.height*grid for c in configs]
            select = m.new_int_var(0, len(configs)-1, f"configuration_{index}")
            width = m.new_int_var(min(widths), max(widths), f"width_{index}")
            height = m.new_int_var(min(heights), max(heights), f"height_{index}")
            m.add_element(select, widths, width)
            m.add_element(select, heights, height)
            penalty = [max(0, int(e["font_size"])-c.font_size)*100 + c.lines-1 for c in configs]
            readability = m.new_int_var(min(penalty), max(penalty), f"readability_{index}")
            m.add_element(select, penalty, readability)
            quality_cost.append(readability * max(1, round(b.importance*10)))
            event["configurations"][b.id] = len(configs)
        else:
            if is_locked:frozen.update(b.ids)
            # Cover the true extent rather than rounding its dimensions down.
            # Locked geometry is emitted verbatim; unlocked translated geometry
            # has a maximum .005px position rounding error in either direction.
            widths = [math.ceil(r.x1*grid)-math.floor(r.x*grid)]
            heights = [math.ceil(r.y1*grid)-math.floor(r.y*grid)]
            width, height, select = widths[0], heights[0], None
            if not is_locked:
                width += 1
                height += 1
        if min(widths) > W or min(heights) > H:
            return finish(None, "INFEASIBLE", reason="block cannot fit canvas", ids=b.ids,
                          conflicting_locks=sorted(locks.intersection(b.ids)))
        x = m.new_int_var(0, W-min(widths), f"x_{index}")
        y = m.new_int_var(0, H-min(heights), f"y_{index}")
        m.add(x+width <= W)
        m.add(y+height <= H)
        if is_locked and b.id not in group_choices:
            m.add(x == math.floor(r.x*grid))
            m.add(y == math.floor(r.y*grid))
        if b.id in group_choices:
            for number,config in enumerate(configs):
                if config.anchor is not None:
                    selected_config=m.new_bool_var(f'anchored_{index}_{number}')
                    m.add(select==number).only_enforce_if(selected_config)
                    m.add(select!=number).only_enforce_if(selected_config.Not())
                    m.add(x==math.floor(config.anchor[0]*grid)).only_enforce_if(selected_config)
                    m.add(y==math.floor(config.anchor[1]*grid)).only_enforce_if(selected_config)
        for axis, ref, limit in ((x, r.x, W), (y, r.y, H)):
            distance = m.new_int_var(0, max(1, 2*limit+abs(round(ref*grid))), f"movement_{index}_{len(movement_cost)}")
            m.add_abs_equality(distance, axis-round(ref*grid))
            movement_cost.append(distance*max(1, round(b.importance*10)))
        m.add_hint(x, min(max(0, round(r.x*grid)), W-min(widths)))
        m.add_hint(y, min(max(0, round(r.y*grid)), H-min(heights)))
        variables[b.id] = x, y, width, height, select

    disjoint = []
    for index, (a, _) in enumerate(groups):
        for b, _ in groups[index+1:]:
            old_a, old_b = source_groups.get(a.id), source_groups.get(b.id)
            both_text = all(original[eid]["type"] == "text" for eid in a.ids+b.ids if eid in original)
            overlay = old_a is not None and old_b is not None and old_a.intersection_area(old_b) > 0 and not both_text
            if overlay:
                continue
            ax, ay, aw, ah, _ = variables[a.id]
            bx, by, bw, bh, _ = variables[b.id]
            flags = [m.new_bool_var(f"separate_{index}_{b.id}_{k}") for k in range(4)]
            for condition, flag in zip((ax+aw+grid <= bx, bx+bw+grid <= ax,
                                         ay+ah+grid <= by, by+bh+grid <= ay), flags):
                m.add(condition).only_enforce_if(flag)
            m.add_bool_or(flags)
            disjoint.append((a.id, b.id))
    invalid = m.validate()
    if invalid:
        return finish(None, "MODEL_INVALID", reason=invalid)
    if cancelled is not None and cancelled():
        return finish(None, "CANCELLED")
    remaining = budget-(time.monotonic()-start)
    if remaining <= 0:
        return finish(None, "UNKNOWN", reason="budget exhausted during construction")
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 7
    quality = sum(quality_cost)
    movement = sum(movement_cost)
    m.minimize(quality if quality_cost else movement)
    solver.parameters.max_time_in_seconds = remaining
    stop = threading.Event()
    watcher = None
    if cancelled is not None:
        def watch():
            while not stop.wait(.02):
                if cancelled():
                    solver.stop_search()
                    break
        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
    try:
        status = solver.solve(m)
        if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
            return finish(None, "CANCELLED" if cancelled is not None and cancelled() else solver.status_name(status),
                          conflicting_locks=sorted(locks) if status == cp_model.INFEASIBLE else [])
        # Save the first feasible solution: the second pass may time out without
        # any incumbent. Freeze its achieved quality even when not proven best.
        snapshot = {key: (solver.value(x), solver.value(y), solver.value(sel) if sel is not None else None)
                    for key, (x, y, _, _, sel) in variables.items()}
        priority_proven = status == cp_model.OPTIMAL
        quality_value = solver.value(quality) if quality_cost else 0
        remaining = budget-(time.monotonic()-start)
        if quality_cost and remaining > .005 and not (cancelled is not None and cancelled()):
            m.add(quality <= quality_value)
            m.minimize(movement)
            solver.parameters.max_time_in_seconds = remaining
            second = solver.solve(m)
            if second in (cp_model.FEASIBLE, cp_model.OPTIMAL):
                status = second
                snapshot = {key: (solver.value(x), solver.value(y), solver.value(sel) if sel is not None else None)
                            for key, (x, y, _, _, sel) in variables.items()}
        event["readability_optimal"] = priority_proven
    finally:
        stop.set()
        if watcher is not None:
            watcher.join(timeout=.05)
    out = copy.deepcopy(design)
    by = {e["id"]: e for e in out["elements"]}
    selected = {}
    for b, r in groups:
        px, py, choice = snapshot[b.id]
        if set(b.ids).intersection(frozen):
            continue
        if choice is not None:
            c = choices[b.id][choice]
            if b.id in group_choices:
                for eid,patch in c.place(px/grid+.005,py/grid+.005).items():by[eid].update(patch)
                selected[b.id]={'configuration_id':c.id,'policy_id':c.group_id,'policy_version':c.policy_version,
                                'width':c.width,'height':c.height,'anchored':c.anchor is not None}
            else:
                by[b.ids[0]].update(x=px/grid, y=py/grid, width=c.width, height=c.height, font_size=c.font_size)
                selected[b.id] = {"font_size": c.font_size, "width": c.width, "height": c.height,
                                  "lines": c.lines, "profile": c.typography_profile}
        else:
            dx, dy = px/grid-r.x+.005, py/grid-r.y+.005
            for eid in b.ids:
                # One common translation preserves fractional rigid-group
                # offsets. Independently rounding each child would shear it.
                by[eid]["x"] = by[eid]["x"]+dx
                by[eid]["y"] = by[eid]["y"]+dy
    # Verify the actual rounded materialization, not just the mathematical model.
    rectangles = {b.id: r for b, r in units(A, out)}
    for eid,rectangle in pins.items():
        if any(abs(by[eid].get(key,-1e9)-value)>1e-6 for key,value in rectangle.items()):
            return finish(None,'MATERIALIZATION_INVALID',reason='child lock changed',ids=[eid])
    for b, r in units(A, out):
        if r.x < -1e-6 or r.y < -1e-6 or r.x1 > cw+1e-6 or r.y1 > ch+1e-6:
            return finish(None, "MATERIALIZATION_INVALID", reason="rounded bounds", ids=b.ids)
        for eid in b.ids:
            e = by[eid]
            if eid in frozen and e != original[eid]:
                return finish(None, "MATERIALIZATION_INVALID", reason="lock changed", ids=[eid])
            if e["type"] == "text":
                f = measure(e, e["font_size"], e["width"])
                if f.overflow or f.height > e["height"]+.5:
                    return finish(None, "MATERIALIZATION_INVALID", reason="text overflow", ids=[eid])
    if any(rectangles[a].intersection_area(rectangles[b]) > 1e-6 for a, b in disjoint):
        return finish(None, "MATERIALIZATION_INVALID", reason="rounded collision")
    return finish(out, solver.status_name(status), geometry_verified=True,
                  selected=selected, readability_penalty=quality_value,
                  cancellation_observed=cancelled is not None and bool(cancelled()))
