# Synonym Resolution

BADA3 and BADA4 backends use synonym files to map ICAO aircraft type codes
to data files on disk. The resolution functions handle normalization,
synonym lookup, filesystem fallback, and error reporting.

## BADA4

::: openleaf.performance.models.bada4.synonyms.resolve_bada4_file

Resolves an ICAO code to a BADA4 XML file path. Resolution order:

1. Synonym lookup: folder/name.xml
2. Synonym lookup: folder/*.xml (any XML in folder)
3. Synonym lookup: data_path/name.xml (direct file)
4. Filesystem scan: folders matching ICAO prefix
5. Filesystem scan: XML files matching ICAO prefix

## BADA3

::: openleaf.performance.models.bada3.synonyms.resolve_bada3_file

Resolves an ICAO code to a BADA3 OPF file path. Resolution order:

1. Synonym lookup: prefix.OPF (exact)
2. Synonym lookup: prefix with underscore variants (`__`, `___`, `_`)
3. Filesystem scan: ICAO + suffix patterns (`__`, `___`, `_`, empty)
