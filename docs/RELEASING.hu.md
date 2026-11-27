# Verzió kiadása

Hogyan készül egy verzió (a karbantartónak)

1. **Verzió.** Írd át a `version` értékét a gyökér `Cargo.toml`-ban és a `TAG`-et az `index.html`-ben az új számra (például `0.1.0-alpha.12` és `v0.1.0-alpha.12`), futtasd a `cargo update -w --offline` parancsot, add hozzá a `docs/releases/<tag>.md` fájlt, majd commit és push a `main` ágra.
2. **Építés.** Várd meg, hogy az adott commit `build` és `engine-ci` futása zöld legyen. Ezután indítsd kézzel a `build` workflow-t (Actions, Run workflow), és add meg a címkét, például `v0.1.0-alpha.12`. Minden platformra épít, és **piszkozat** kiadást hoz létre.
3. **Közzététel.** Nyisd meg a piszkozatot, másold a `docs/releases/<tag>.md` tartalmát a leírásba, jelöld be az előzetes kiadást, és nyomd meg a Publish gombot. **Ne változtasd meg a címke nevét a piszkozatban**: a fájlok a `Cargo.toml` szerinti verzióval épültek, más címkénél a program egy verziót jelent, a kiadás pedig másikat (és a frissítésellenőrzés végtelenül frissítést kínál). A workflow elutasít egy olyan címkét, amely már másik commitra mutat.

## Ha tévedésből közzétettek egy számot

Ne használd újra. Töröld kézzel azt a kiadást és a címkéjét, és add ki a következő számot.

## A kiadást soha ne hozd létre kézzel

A `build` workflow maga hozza létre a piszkozatot, minden fájllal csatolva. Az azonos címkével kézzel létrehozott kiadás (piszkozatként is) üres, és elbuktatja az utolsó lépést. Ha megtörtént: töröld az üres kiadást (a címkét tartsd meg), és a `build` futtatásán használd a **Re-run failed jobs** gombot.
