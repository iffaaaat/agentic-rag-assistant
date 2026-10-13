import re


def extract_cited_evidence(answer, context):
    """Extract retrieved source chunks cited in an answer."""

    cited_numbers = set(
        re.findall(r"\[Source (\d+)\]", answer)
    )

    source_matches = list(
        re.finditer(r"\[Source (\d+)\]:", context)
    )

    source_map = {}

    for index, match in enumerate(source_matches):
        source_number = match.group(1)
        start = match.end()

        if index + 1 < len(source_matches):
            end = source_matches[index + 1].start()
        else:
            end = len(context)

        source_map[source_number] = context[start:end].strip()

    cited_evidence = []

    for number in sorted(cited_numbers, key=int):
        if number in source_map:
            cited_evidence.append(
                f"[Source {number}]: {source_map[number]}"
            )

    return "\n\n".join(cited_evidence)


def validate_citations(answer, context):
    """Check whether cited source numbers exist in retrieved context."""

    available_sources = set(
        re.findall(r"\[Source (\d+)\]:", context)
    )

    cited_sources = set(
        re.findall(r"\[Source (\d+)\]", answer)
    )

    invalid_sources = cited_sources - available_sources

    return {
        "valid": bool(cited_sources) and not invalid_sources,
        "cited_sources": sorted(cited_sources, key=int),
        "invalid_sources": sorted(invalid_sources, key=int),
    }
