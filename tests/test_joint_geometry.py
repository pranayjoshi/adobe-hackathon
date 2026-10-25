"""Behavioral checks for renderer-compatible joint sizing and placement."""
import copy
from types import SimpleNamespace
import unittest

from engine.configurations import enumerate_text_configurations
from engine.constraints import solve, solve_joint
from engine.typography import measure, measurement_identity, renderer_available


def text(eid="headline", **overrides):
    return {**dict(id=eid, type="text", content="Hello wonderful world", x=5, y=5,
                width=60, height=100, font_family="Solstice Sans", font_size=20,
                color="rgba(0,0,0,1)"), **overrides}


def analysis(*ids):
    blocks = [SimpleNamespace(id=str(i), ids=list(group), role="headline", importance=1)
              for i, group in enumerate(ids)]
    return SimpleNamespace(all_blocks=lambda: blocks)


@unittest.skipUnless(renderer_available(), "local renderer/font resources unavailable")
class JointGeometryTests(unittest.TestCase):
    def test_configurations_match_actual_renderer_blobs_and_pixel_bounds(self):
        import skia
        from PIL import Image
        from cr_renderer.schema import TextElement
        from cr_renderer.text_utils import generate_map, make_text_blob, make_text_spans, render_text
        from engine.typography import _FM
        from renderer.converter.convert import convert_chameleon_to_crello
        examples = [text(), text(style_runs=[{"start": 6, "end": 14, "font_weight": "bold"}]),
                    text(letter_spacing=1.25, style_runs=[{"start": 6, "end": 14, "font_weight": "bold"}]),
                    text(content="UnbreakableSuperword", letter_spacing=.5)]
        for el in examples:
            with self.subTest(el=el):
                before = copy.deepcopy(el)
                configs = enumerate_text_configurations(el, widths=[180, 280], max_width=280,
                    max_height=180, fs_min=20, max_configs=12, allow_estimated=False)
                self.assertTrue(configs)
                self.assertEqual(before, el)
                for config in configs:
                    fitted = {**el, "width": config.width, "height": config.height, "font_size": config.font_size}
                    converted, _ = convert_chameleon_to_crello({"canvas": {"width": 300, "height": 200}, "elements": [fitted]}, _FM)
                    native = TextElement(uuid="", type="textElement", width=config.width,
                        height=config.height, text=converted["text"][0], fontSize=config.font_size,
                        font=fitted["font_family"], lineHeight=converted["line_height"][0],
                        textAlign="left", capitalize=converted["capitalize"][0],
                        letterSpacing=converted["letter_spacing"][0],
                        boldMap=generate_map(converted["font_bold"][0]),
                        italicMap=generate_map(converted["font_italic"][0]),
                        colorMap=generate_map(converted["text_color"][0]),
                        lineMap=generate_map(converted["text_line"][0]))
                    lines = list(make_text_spans(native))
                    advances = [sum(make_text_blob(native, span, _FM).width for span in line) for line in lines]
                    self.assertEqual(config.lines, len(lines))
                    self.assertAlmostEqual(config.widest_advance, max(advances), delta=.01)
                    surface = skia.Surface(300, 200)
                    canvas = surface.getCanvas()
                    canvas.clear(skia.ColorWHITE)
                    canvas.translate(5, 5)
                    render_text(canvas, _FM, native)
                    rendered = Image.fromarray(surface.makeImageSnapshot().toarray()).convert("RGB")
                    from PIL import ImageChops
                    bounds = ImageChops.difference(rendered, Image.new("RGB", rendered.size, "white")).getbbox()
                    self.assertIsNotNone(bounds)
                    # Raster antialiasing and glyph bearings can extend by 1px.
                    self.assertLessEqual(bounds[2], 5+config.width+2)
                    self.assertLessEqual(bounds[3], 5+config.height+2)

    def test_joint_fit_changes_configuration_and_preserves_lock_and_content(self):
        headline = text()
        tag = {**text("tag"), "content": "OK", "x": 250, "y": 5, "width": 45,
               "height": 20, "font_size": 10}
        source = {"canvas": {"width": 400, "height": 200}, "elements": [headline, tag]}
        design = copy.deepcopy(source)
        design["canvas"] = {"width": 200, "height": 60}
        design["elements"][1].update(x=150, y=38)
        before = copy.deepcopy(design)
        A = analysis(["headline"], ["tag"])
        self.assertIsNone(solve(source, design, A, ["tag"])[0])
        out, event = solve_joint(source, design, A, ["tag"], budget_seconds=2)
        self.assertIsNotNone(out, event)
        self.assertTrue(event["geometry_verified"])
        self.assertEqual(out["elements"][1], before["elements"][1])
        self.assertGreater(out["elements"][0]["width"], headline["width"])
        self.assertGreaterEqual(out["elements"][0]["font_size"], 18)
        self.assertEqual(design, before)
        for original, result in zip(source["elements"], out["elements"]):
            for field in ("content", "font_family", "color"):
                self.assertEqual(original[field], result[field])
            fit = measure(result, result["font_size"], result["width"])
            self.assertFalse(fit.overflow)
            self.assertLessEqual(fit.height, result["height"]+.5)

    def test_conflicting_locks_infeasible_and_no_budget_unknown(self):
        source = {"canvas": {"width": 300, "height": 100}, "elements": [text("a"), {**text("b"), "x": 200}]}
        design = copy.deepcopy(source)
        design["elements"][1]["x"] = 5
        A = analysis(["a"], ["b"])
        out, event = solve_joint(source, design, A, ["a", "b"], budget_seconds=1)
        self.assertIsNone(out)
        self.assertEqual(event["status"], "INFEASIBLE")
        self.assertEqual(event["conflicting_locks"], ["a", "b"])
        self.assertEqual(solve_joint(source, design, A, budget_seconds=0)[1]["status"], "UNKNOWN")
        self.assertEqual(solve_joint(source, design, A, cancelled=lambda: True)[1]["status"], "CANCELLED")

    def test_missing_face_is_explicitly_unverified(self):
        el = {**text(), "font_family": "definitely absent family"}
        self.assertFalse(measurement_identity(el)["verified"])
        self.assertFalse(enumerate_text_configurations(el, allow_estimated=False))
        estimated = enumerate_text_configurations(el, max_width=300)
        self.assertTrue(estimated)
        self.assertFalse(any(c.verified for c in estimated))

    def test_rigid_rotated_group_keeps_fractional_internal_offsets(self):
        members = [dict(id="a", type="shape", x=5.123, y=5.456, width=30, height=20, rotation=15),
                   dict(id="b", type="shape", x=40.789, y=7.654, width=10, height=10)]
        source = {"canvas": {"width": 100, "height": 100}, "elements": members}
        design = copy.deepcopy(source)
        for el in design["elements"]:
            el["x"] += 80
        out, event = solve_joint(source, design, analysis(["a", "b"]), budget_seconds=1)
        self.assertIsNotNone(out, event)
        for field in ("x", "y"):
            self.assertAlmostEqual(out["elements"][1][field]-out["elements"][0][field],
                                   members[1][field]-members[0][field], places=10)

    def test_policy_joint_reflows_stack_with_exact_child_lock(self):
        from engine.analysis import Analysis
        from engine.blocks import TextBlock,StackBlock
        from engine.group_policies import GroupPolicy
        from engine.configurations import enumerate_group_configurations
        a={**text('a'),'content':'Hello','font_size':18,'x':10,'y':10,'width':60,'height':30}
        b={**a,'id':'b','content':'World','y':50}
        source={'canvas':{'width':200,'height':200},'elements':[a,b]}
        policy=GroupPolicy('stack',('a','b'),None,'inferred','reflowable',('y','x'),('a','b'),
            'left',None,'test measured display stack',kind='display_stack')
        children=[TextBlock(a,'headline','Headline',1),TextBlock(b,'subhead','Subhead',.7)]
        stack=StackBlock(children,'stack','headline','Offer',1,1,policy=policy)
        A=Analysis(200,200,blocks=[stack],roles={'a':'headline','b':'subhead'},policies=[policy],group_mode='hierarchical')
        target={'width':200,'height':35};design=copy.deepcopy(source);design['canvas']=target
        design['elements'][0].update(x=10,y=2)
        pins={'a':{'x':10.0,'y':2.0,'width':60.0,'height':30.0}}
        self.assertIsNone(solve(source,design,A,pins)[0])
        configs=enumerate_group_configurations(policy,source,A,target,locks=pins)
        self.assertTrue(configs)
        out,event=solve_joint(source,design,A,pins,budget_seconds=2)
        self.assertIsNotNone(out,event);self.assertTrue(event['geometry_verified'])
        elements={e['id']:e for e in out['elements']}
        for key,value in pins['a'].items():self.assertAlmostEqual(elements['a'][key],value,places=7)
        self.assertGreater(elements['b']['x'],elements['a']['x']+elements['a']['width'])
        self.assertEqual(elements['a']['content'],'Hello');self.assertEqual(elements['b']['content'],'World')
        self.assertTrue(event['selected']['stack']['configuration_id'])
        self.assertEqual(event['selected']['stack']['policy_version'],'group-policy/1')
        # Two mutually inconsistent child anchors cannot be silently relaxed.
        impossible={**pins,'b':{'x':10.0,'y':2.0,'width':60.0,'height':30.0}}
        result,status=solve_joint(source,design,A,impossible,budget_seconds=2)
        self.assertIsNone(result);self.assertEqual(status['status'],'INFEASIBLE')


if __name__ == "__main__":
    unittest.main()
