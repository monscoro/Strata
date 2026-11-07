# Dave-Diskussion

Die Unterhaltung mit Dave über den 4-bit-KV-Cache, in der Reihenfolge des Threads:

| Datei | Inhalt |
|---|---|
| `email_to_dave.txt` | Unsere erste Mail: Plateau, q4lm8v, die Messungen und die eine Frage. |
| `daves_reply.md` | Daves erste Empfehlung (L3, per-channel K-Skala vor der Hadamard-Rotation) und sein Kill-Design. |
| `daves_survey.md` | Daves Literatur-Survey zu L3 (KVQuant, KVarN, KVLinC, OSCAR). |
| `email_to_dave_reply.txt` | Unsere Antwort auf die Revision: Skalenteil gemessen, Normkorrektur offen, P0/P1/P2. |
| `daves_second_reply.md` | Daves Korrektur (L1 ist nicht tot, `turboquant_4bit_nc` ist 16/16) und die Antworten zur Spezifikation. |
| `email_to_dave_thanks.txt` | Unsere Dankesmail: L1/16/16 akzeptiert, P0a gemessen, P0b als radiale/tangentiale Zerlegung angekündigt. |
| `daves_thanks_reply.md` | Daves dritte Antwort: keine p99-Zahl in der Literatur, der Qwen-Worst-Case-Kollaps (min. Cosinus) und die Ergänzungen zu P0b. |
| `PLAN.md` | Der Plan nach der zweiten Antwort: P0a-Ergebnis, P0b-Capture, Arme und die Entscheidungsregel vor dem Lauf, mit Nachtrag zur dritten Antwort. |

Das ist Arbeitskorrespondenz, keine veröffentlichten Docs. Die Messzahlen stehen in
`bench/results/`; die Formatfakten in `include/strata/kernels/kv_q4.hpp`.
