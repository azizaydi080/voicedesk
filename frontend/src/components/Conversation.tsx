import type { Turn } from "@/lib/protocol";

const LANG_LABEL = { ar: "عربي", fr: "FR", en: "EN" } as const;

export function Conversation({ turns, partial }: { turns: Turn[]; partial: string }) {
  if (!turns.length && !partial) {
    return (
      <div className="empty">
        <p>Try, in any of the three languages:</p>
        <ul>
          <li dir="auto">“I&apos;d like a dentist appointment tomorrow morning.”</li>
          <li dir="auto">« Je voudrais annuler mon rendez-vous 1002. »</li>
          <li dir="auto">«وقتاش تحلّو العيادة نهار السبت؟»</li>
        </ul>
      </div>
    );
  }
  return (
    <ol className="conversation">
      {turns.map((t) => (
        <li key={t.id} className="exchange">
          <div className="bubble user" dir="auto">
            {t.language && (
              <span className="badge" title={`detected by ${t.languageMethod}`}>
                {LANG_LABEL[t.language]}
              </span>
            )}
            {t.user}
          </div>
          {(t.agent || t.tools.length > 0) && (
            <div className="bubble agent" dir="auto">
              {t.tools.map((x) => (
                <span key={x.id} className={`chip ${x.ok === false ? "chip-err" : ""}`}>
                  ⚙ {x.name}
                </span>
              ))}
              {t.agent}
              {t.interrupted && <span className="interrupted">interrupted</span>}
            </div>
          )}
        </li>
      ))}
      {partial && (
        <li className="exchange">
          <div className="bubble user partial" dir="auto">{partial}</div>
        </li>
      )}
    </ol>
  );
}
