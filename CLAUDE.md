# SimForge / JaxRLWorld Development Guide

## Language
- 대화: 한국어, 코드/커밋/주석: 영어

## Response Format
- MUST: 파일을 수정했으면 답변 마지막에 수정된 파일 경로 목록을 보고

## Code Style
- NEVER: 함수 내부에서 import 하지 말 것. 유일한 예외: 순환 참조(circular import)
- NEVER: silent fallback. 실패하면 즉시 crash. `except: pass`나 빈 fallback 금지
- MUST: top-level import를 제거하기 전에 grep으로 re-export 여부 확인

## Architecture Principles
- NEVER: 원본 동작에 없던 기능 추가. 리팩토링은 동작 보존이 원칙
- MUST: dead code 제거 전 grep으로 caller 전수 조사 (legacy preset 포함)
- 시뮬레이터를 다루다가 헷갈리는 부분은 Genesis/, Newton/, Mjlab/ 폴더를 들어가서 직접 탐색하고 이해한 다음 구현

## Git / Workflow
- 커밋 메시지는 descriptive하게, 미래 참조에 유용하도록
- MonoRepo: `SimForge/` 안에 `JaxRLWorld/` (git tracked), 시뮬레이터들(`Genesis/`, `Newton/`, `Mjlab/` 등)은 `.gitignore`로 제외
- `.gitignore`에서 시뮬레이터 경로는 root-relative (`/Genesis/`, `/Newton/`) -- macOS case-insensitive 이슈 방지
