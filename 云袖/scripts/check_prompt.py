#!/usr/bin/env python3
"""Validate explicitly bounded Yunxiu prompt blocks. Python standard library only."""
import argparse
import hashlib
import json
import re
from pathlib import Path

PREFIX = ['视觉基调', '风格核心', '人物细节', '运镜规则']
FIELDS = ['素材引用', '轴线控制', '前情提要', '人物站位', '剧情描述']
TOKEN = re.compile(r'<!--\s*(yunxiu-prompt|/yunxiu-prompt)\b([^>]*)-->')
SHOT = re.compile(r'^镜头\s*(\d+)(?:（[^）]*）|\([^)]*\))?\s*[:：](.*)$')


def clean(line):
    line = re.sub(r'^\s*#{1,6}\s+', '', line).strip()
    return line.replace('**', '').strip()


def read_prefix(framework):
    text = framework.read_text(encoding='utf-8-sig')
    fence = re.search(r'```text\s*\n(.*?)```', text, re.S)
    if not fence:
        raise ValueError('权威框架缺少 text 代码块')
    lines = [x.strip() for x in fence[1].splitlines() if x.strip()]
    if len(lines) < 4 or any(not lines[i].startswith(k + '：') for i, k in enumerate(PREFIX)):
        raise ValueError('权威框架前四行不是四条固定基调')
    return lines[:4]


def validate(text, prefix, expected_ids=None, os_ids=()):
    errors, results, spans = [], [], []
    active = None
    ids = []
    for token in TOKEN.finditer(text):
        if token[1] == 'yunxiu-prompt':
            if active:
                errors.append('正文标记嵌套或缺少结束标记')
            match = re.fullmatch(r'\s+id="([A-Za-z0-9_-]+)"\s*', token[2])
            sid = match[1] if match else '?'
            if not match:
                errors.append('开始标记必须含唯一 id，例如 C12-01')
            active = (sid, token.end(), token.start())
        else:
            if active is None:
                errors.append('结束标记没有对应开始标记')
                continue
            sid, start, marker_start = active
            active = None
            ids.append(sid)
            spans.append((marker_start, token.end()))
            lines = [clean(x) for x in text[start:token.start()].splitlines() if x.strip()]
            local = []
            if lines[:4] != prefix:
                local.append('四条开场基调缺失、顺序错误或未逐字沿用权威框架')
            headings = []
            for i, line in enumerate(lines):
                m = re.match(r'^(素材引用|轴线控制|前情提要|人物站位|剧情描述|内心独白音效)\s*[:：](.*)$', line)
                if m:
                    headings.append((i, m[1], m[2]))
            names = [h[1] for h in headings]
            want = FIELDS + (['内心独白音效'] if '内心独白音效' in names else [])
            if names != want:
                local.append('栏目须各出现一次且按素材引用→轴线控制→前情提要→人物站位→剧情描述→可选内心独白音效排列')
            shots = [(i, SHOT.match(line)) for i, line in enumerate(lines) if SHOT.match(line)]
            first_shot = shots[0][0] if shots else len(lines)
            if not shots:
                local.append('缺少逐镜正文（镜头1：……）')
            if headings and (headings[0][0] != 4 or headings[-1][0] >= first_shot):
                local.append('四条基调后应直接进入素材引用；全部栏目须位于镜头之前')
            for n, (i, name, inline) in enumerate(headings):
                end = headings[n + 1][0] if n + 1 < len(headings) else first_shot
                content = '\n'.join([inline] + lines[i + 1:end]).strip()
                if not content or content in ('待填写', '……', '...', '无则删除本行'):
                    local.append(name + '为空或仍是空模板')
                if name == '内心独白音效':
                    if '不做口型' not in content or '镜' not in content:
                        local.append('内心独白音效须标注不做口型和覆盖镜号；原句完整性仍需人工核对')
            if sid in os_ids and '内心独白音效' not in names:
                local.append('原文发声清单指定本段含内心独白，但正文缺少该栏目')
            numbers = [int(m[1]) for _, m in shots]
            if numbers != list(range(1, len(numbers) + 1)):
                local.append('镜号必须从1起连续，不重复、不跳号')
            for n, (i, m) in enumerate(shots):
                end = shots[n + 1][0] if n + 1 < len(shots) else len(lines)
                content = '\n'.join([m[2]] + lines[i + 1:end]).strip()
                if not content or content in ('待填写', '……', '...'):
                    local.append('镜头' + m[1] + '为空')
            # Catch another prefix or a misspelled/legacy shot heading inside the same block.
            for i, line in enumerate(lines[4:], 4):
                if any(line.startswith(k + '：') for k in PREFIX):
                    local.append('正文中重复出现开场基调；每段须单独标记')
                if re.match(r'^镜头\s*\d+', line) and not SHOT.match(line):
                    local.append('镜头标题格式无效：' + line[:40])
            results.append({'id': sid, 'shots': len(shots), 'passed': not local, 'errors': local})
            errors.extend(sid + ': ' + e for e in local)
    if active:
        errors.append('最后一个正文块没有结束标记')
    if not ids:
        errors.append('未找到独立正文块；不得把自由分镜或无正文的文件标为云袖格式通过')
    if len(set(ids)) != len(ids):
        errors.append('正文段ID重复')
    if expected_ids is not None and ids != list(expected_ids):
        errors.append('正文段ID或顺序与逐段计划指定范围不符')
    if set(os_ids) - set(ids):
        errors.append('内心独白检查范围包含不存在的段ID')
    outside, last = [], 0
    for a, b in spans:
        outside.append(text[last:a])
        last = b
    outside.append(text[last:])
    if any(re.match(r'^(视觉基调\s*[:：]|镜头\s*\d+)', clean(x)) for x in '\n'.join(outside).splitlines()):
        errors.append('正文块外仍有提示词或镜头，存在漏检；将所有交付正文纳入标记')
    return {'passed': not errors, 'segments': results, 'errors': errors,
            'scope': '仅结构与固定前缀；不证明原文覆盖、语义、表演、旁白校时、真实素材绑定或实片通过'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('draft', type=Path)
    parser.add_argument('--framework', type=Path, required=True)
    parser.add_argument('--segments', nargs='+', required=True, help='按逐段计划列出本次交付的全部段ID')
    parser.add_argument('--os-segments', nargs='*', default=[])
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--prefix-overrides', type=Path, help='仅用于用户明确覆盖四条基调；含source与lines字典')
    args = parser.parse_args()
    # Reports must never overwrite input, source rules, or the checker itself.
    inputs = [args.draft, args.framework, Path(__file__)]
    if args.prefix_overrides:
        inputs.append(args.prefix_overrides)
    if args.report.resolve() in [x.resolve() for x in inputs]:
        parser.error('报告路径不得覆盖输入文件')
    try:
        prefix = read_prefix(args.framework)
        override = None
        if args.prefix_overrides:
            override = json.loads(args.prefix_overrides.read_text(encoding='utf-8-sig'))
            if not isinstance(override, dict) or not isinstance(override.get('source'), str) or not override['source'].strip():
                raise ValueError('覆盖文件须有用户原话及来源source')
            lines = override.get('lines')
            if not isinstance(lines, dict) or not lines or set(lines) - set(PREFIX):
                raise ValueError('覆盖范围仅限四条固定基调')
            for key, value in lines.items():
                if not isinstance(value, str) or not value.startswith(key + '：') or '\n' in value or '\r' in value:
                    raise ValueError('覆盖值须是含原栏目名的完整单行')
                prefix[PREFIX.index(key)] = value
        raw = args.draft.read_bytes()
        report = validate(raw.decode('utf-8-sig'), prefix, args.segments, args.os_segments)
        report.update({'draft': str(args.draft.resolve()), 'draft_sha256': hashlib.sha256(raw).hexdigest(),
                       'framework_sha256': hashlib.sha256(args.framework.read_bytes()).hexdigest(),
                       'user_prefix_override': override})
    except (OSError, ValueError, UnicodeError) as exc:
        report = {'passed': False, 'errors': [str(exc)]}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(('PASS' if report['passed'] else 'FAIL') + ': ' + str(args.report))
    for error in report.get('errors', []):
        print('- ' + error)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
