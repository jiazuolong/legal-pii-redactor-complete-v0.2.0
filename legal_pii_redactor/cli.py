"""CLI entry point: legal-redact command."""

import argparse
import sys
from pathlib import Path

from legal_pii_redactor.entities import EntityType
from legal_pii_redactor.pipeline import LegalPIIRedactor


def main():
    parser = argparse.ArgumentParser(
        description="中文法律文书PII脱敏工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  legal-redact input.txt                           # 脱敏并输出到stdout
  legal-redact input.txt -o output.txt             # 输出到文件
  legal-redact input.txt --style mask              # 用***替换
  legal-redact input.txt --detect-only             # 只检测不替换
  legal-redact --text "被告人张三，身份证号110101199001011234"
  legal-redact input.txt --no-ner                  # 不使用NER模型（更快）
  legal-redact input.txt --types PERSON_NAME ID_CARD PHONE  # 只脱敏指定类型
  legal-redact input.txt --exclude-types DATE MONEY         # 排除日期和金额
""",
    )
    parser.add_argument("file", nargs="?", help="输入文件路径")
    parser.add_argument("--text", "-t", help="直接输入文本（替代文件）")
    parser.add_argument("--output", "-o", help="输出文件路径（默认stdout）")
    parser.add_argument(
        "--style", "-s",
        choices=["placeholder", "mask", "delete"],
        default="placeholder",
        help="替换风格: placeholder=[姓名1], mask=***, delete=删除 (默认: placeholder)",
    )
    parser.add_argument("--detect-only", "-d", action="store_true", help="只检测实体，不替换")
    parser.add_argument("--no-ner", action="store_true", help="禁用NER模型（更快，但人名召回降低）")
    parser.add_argument(
        "--types",
        nargs="+",
        choices=[e.value for e in EntityType],
        help="只处理指定的实体类型",
    )
    parser.add_argument(
        "--exclude-types",
        nargs="+",
        choices=[e.value for e in EntityType],
        dest="exclude_types",
        help="排除指定的实体类型（与 --types 互斥，优先级低于 --types）",
    )

    args = parser.parse_args()

    # Get input text
    if args.text:
        text = args.text
    elif args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"错误: 文件不存在: {path}", file=sys.stderr)
            sys.exit(1)
        text = path.read_text(encoding="utf-8")
    else:
        # Read from stdin
        if sys.stdin.isatty():
            parser.print_help()
            sys.exit(0)
        text = sys.stdin.read()

    # Initialize
    redactor = LegalPIIRedactor(use_ner=not args.no_ner)

    # Entity type filter
    type_filter = [EntityType(t) for t in args.types] if args.types else None
    exclude_filter = [EntityType(t) for t in args.exclude_types] if args.exclude_types else None

    if args.detect_only:
        entities = redactor.detect(text, entity_types=type_filter, exclude_types=exclude_filter)
        for e in entities:
            print(f"[{e.start:5d}:{e.end:5d}]  {e.entity_type.value:15s}  \"{e.text}\"")
        print(f"\n共检测到 {len(entities)} 个PII实体")
    else:
        result = redactor.redact(text, replacement_style=args.style, entity_types=type_filter, exclude_types=exclude_filter)
        if args.output:
            Path(args.output).write_text(result, encoding="utf-8")
            print(f"已保存到: {args.output}", file=sys.stderr)
        else:
            print(result)


if __name__ == "__main__":
    main()
