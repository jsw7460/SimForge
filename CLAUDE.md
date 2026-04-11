# SimForge / JaxRLWorld Development Guide

## Language
- 대화: 한국어, 코드/커밋/주석: 영어
- 파일 수정 시 답변 마지막에는 수정된 파일의 이름과 경로를 반드시 보고

## Code Style

### Imports
- **import는 파일 상단에.** 함수 내부 import 금지. 유일한 예외: 순환 참조(circular import).

### Error Handling
- **silent fallback 금지.** 실패하면 즉시 crash. `except: pass`나 빈 fallback 절대 안됨.

## Architecture Principles

### 리팩토링 시 주의사항
- **원본 동작에 없던 기능 추가 금지.** 예: Newton/Genesis에 없던 joint limit clamping을 common 버전에 추가하면 안됨. 원본과 동일한 동작만.
- **3개 sim 전부 확인.** cross-sim 변경 시 Newton, Genesis, MuJoCo 모두에서 shape/API 호환성 확인.

## Git / Workflow
- 커밋 메시지는 descriptive하게, 미래 참조에 유용하도록
- MonoRepo: `SimForge/` 안에 `JaxRLWorld/` (git tracked), 시뮬레이터들(`Genesis/`, `Newton/`, `Mjlab/` 등)은 `.gitignore`로 제외
- `.gitignore`에서 시뮬레이터 경로는 root-relative (`/Genesis/`, `/Newton/`) -- macOS case-insensitive 이슈 방지
