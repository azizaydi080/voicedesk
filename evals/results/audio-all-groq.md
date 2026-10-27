# Eval run 20261027-165522

mode `audio` · stage `all` · stt `openai_stt:whisper-large-v3-turbo` · llm `openai_compat:openai/gpt-oss-20b`

| metric | value |
|---|---|
| scenarios | 18 |
| turns | 23 |
| language_id_accuracy | 100% |
| stt_cer_mean | 3% |
| stt_ms_p50 | 332.3 |
| tool_call_accuracy | 96% |
| faq_answer_accuracy | 100% |
| reply_language_accuracy | 100% |
| task_success_rate | 94% |
| errors | 0 |
| llm_ttft_ms_p50 | 425.8 |
| agent_turn_ms_p50 | 10511.4 |
| llm_ttft_ms_p90 | 612.1 |
| rate_limit_wait_s | 257.0 |

| group | turns | language_id | cer | tool_calls | task_success |
|---|---|---|---|---|---|
| en | 6 | 100% | 4% | 100% | 100% |
| fr | 5 | 100% | 1% | 100% | 100% |
| ar | 4 | 100% | 6% | 100% | 100% |
| tn | 4 | 100% | 4% | 75% | 67% |
| gulf | 1 | 100% | 0% | 100% | 100% |
| mixed | 1 | 100% | 11% | 100% | 100% |
| switch | 2 | 100% | 0% | 100% | 100% |

## Failed scenarios

- `tn_book_dentist#0` lang ar/ar tools_ok=True answer_ok=None · find_slots({"date": "2026-10-28", "part_of_day": "afternoon", "specialty": "dentist"}) · reply: 'مواعيد المتاح في غدٍ بعد الظهر: 13:30، 14:00، 14:30، 15:30. ما اسمك ورقم هاتفك؟'
- `tn_book_dentist#1` lang ar/ar tools_ok=False answer_ok=None · book_appointment({"date": "2026-10-28", "patient_name": "سنية التاريكي", "phone": "22334455", "specialty": "dentist", "time": "14:00"}) · reply: 'تم تأكيد موعدك في 28 أكتوبر الساعة 14:00. رقم الحجز هو 1004. نراكم هناك.'
