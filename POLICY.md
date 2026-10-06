# AssetFactory — Policy

Rules for what is written. One line, one rule.

---

## 1. Unnecessary code

### Such code is never written, and any found is deleted on sight, without discussion.

A scenario unreachable through ordinary interaction with the UI elements that exist does not exist.

Write no guard, no refusal wording and no restored state for a scenario that does not exist.

Client-side validation is the filter for a field's contents; nothing downstream re-implements it.

Whatever gets past that validation arrived by hand; the factory owes it nothing.

A route needs no answer for a state in which its form is not rendered.

A state reachable only through a bug in our own code or a broken configuration is never handled.

A bug in the operating system, a bug in the standard library and a hardware failure are not reasons for a check.

Network failure is handled only where the network leaves localhost.

A guard against a caller that does not exist today is unnecessary code.

A defence against a future edit breaking an invariant is unnecessary code.

A fallback for a row the schema forbids is unnecessary code.

A re-check of what the caller already established is unnecessary code.

A tripwire for an ordering or a nesting our own code decides is unnecessary code.

A fallback for the factory's own files or settings being absent or malformed is unnecessary code.

An absence is checked only where our own design produces it.

Our own design produces an absence when something is not written yet, when a vendor creates it on its own schedule, or when another of our own processes claimed and unlinked it.

An absence caused by a third party, or by the owner deleting something by hand, is not handled.

A file our own code wrote a moment earlier is there; no check on it.

Handling stays only for what a real outside actor does: an agent's arbitrary arguments, a vendor process, a model library's own failures (out of memory, crashes), a race between two people or two agents.

A refusal is written only about real state — a version that moved, a job already finished, a model not installed.

Where the only way in is our own mistake, let the code do whatever it naturally does — crash or anything else — with nothing catching it.

A field, parameter, column or constant with no reader is deleted.

A dormant half of a mechanism that does not exist yet is deleted.

Nothing is kept for a use that has not arrived.

An abstraction is written when the second case exists, never before.

Two independent statements of one fact are one too many; derive the second from the first.

When code is spared because one reader uses it, examine the reader.

Ask not whether something uses it, but whether that use is still justified.

Where the only reader's own justification is thin, delete both together.

Logic that is hard to follow is usually logic that is not needed.

Every review finding is triaged before it becomes work.

Triage answers two questions: what it costs if left alone, and which click, message or call reaches it.

A finding is relayed only with its reachability stated.

When a finding and an implementer disagree about a fact, read the code and settle it.

A review that produces no changes is a normal outcome.

---

## 2. Minimal testing

The suite is minimal by intent.

A test is added only when a named future edit is likely to break the behaviour without anyone noticing.

A test earns its place by how easily the behaviour breaks under later change, never by how important the behaviour is.

A behaviour that fails visibly the first time anyone looks needs no test.

A behaviour whose failure is silent and cumulative needs one.

A behaviour that depends on an external format nobody here controls needs one.

An invariant that no single change can violate on its own needs none.

A test that asserts against a mechanism that no longer exists is deleted with the mechanism.

Nothing of our own is mocked.

The only sanctioned stand-in is a vendor model runtime.

A test never downloads model weights.

A test writes through the same entry point production uses, never a path of its own.

A test drives a step directly rather than racing a background loop.

Every fixture is labelled as copied verbatim, constructed from field names on record, or invented.

An invented fixture is never presented as evidence of what a vendor emits.

Review is reading the code and reasoning about it.

Running something is for settling one specific doubt, and the reviewer says which.

A constant or a capability is verified against the tool that can answer for it.

A constant that cannot be measured is labelled documented or guessed.

When a class of bug is found in one vendor, the other adapters are checked immediately.

---

## 3. Comments and texts

### These rules govern every text in the project; the QA log is the one exception, and history, evidence and reasoning are what it is for.

No text carries the reasoning for what it describes.

No text carries a consequence clause.

No text refers to a past event.

Out: "this was measured", "an earlier version", "found in review", "it once cost a night", "used to", "no longer".

A number offered as an argument is an argument; it goes.

Nothing cites a QA log finding or a DESIGN.md section.

A comment asserts only about the code it sits beside.

A comment states what the code does, what it requires, what a caller must not do, and what must run before what.

A comment never states what a vendor emits, what a format contains, how many of something were counted, or what a run showed.

A comment describes a property a reader can check against the code in front of them.

A system prompt states the instruction and nothing else.

A system prompt never names the machinery that catches a failure to follow the instruction.

A tool description is a prompt and follows every rule for one.

Every clause a reader could answer with "but that does not apply here" is deleted.

An alternative action stays where one is needed.

A definition that sharpens the instruction stays.

A prompt sentence that reassures rather than instructs is a defect.

The design document states the present system: no history, no rationale, no evidence, no policy.

Refusal text names the state and the next action.

Refusal text is written for whoever has to act on it.

Every text the factory writes is in English.
