# Biztonsági és robusztussági változtatások

[English](CHANGES-SECURITY.md) · [Română](CHANGES-SECURITY.ro.md) · [Français](CHANGES-SECURITY.fr.md) · [Italiano](CHANGES-SECURITY.it.md) · [Español](CHANGES-SECURITY.es.md) · [Português](CHANGES-SECURITY.pt.md) · [Deutsch](CHANGES-SECURITY.de.md) · [Magyar](CHANGES-SECURITY.hu.md)

További dokumentumok: [ENGINE.md](ENGINE.hu.md) · [QUICKSTART.md](QUICKSTART.hu.md)

Ez az oldal felsorolja, mit talált a motor és a `corbul` adverzális átvizsgálása, és mit tettünk ellene, hogy megítélhesd, mennyire bízhatsz egy funkcióban. Javítások és korlátok listája, nem biztonsági tanúsítvány: semmit sem auditált harmadik fél.

## Szerver

- **Webhelyek közti kérések.** Az állapotot módosító hívások ellenőrzik az `Origin` fejlécet, és JSON `Content-Type` fejlécet követelnek, így egy meglátogatott weboldal nem késztetheti a böngésződet arra, hogy egy helyi `corbul serve` szerverre küldjön. A böngészők csak azokról az eredetekről engedélyezettek, amelyeket a `--cors` kapcsolóval felsorolsz.
- **Admin útvonalak.** A `/api/create`, `/api/delete`, `/models/load` és `/models/unload` a mestertokent igényli. Az API-kulcsok birtokosai csak akkor használhatják őket, ha a szervert az `--allow-admin` kapcsolóval indítod.
- **Titkok a parancssoron kívül.** A `--token-file`, a `--key-file`, a `--receipt-key-file` kapcsoló, valamint a `CORBUL_TOKEN` és a `CORBUL_MESH_KEY` változó távol tartja a titkokat a folyamatlistától. A sima `--token`, `--key` és `--receipt-key` továbbra is működik, és látható a gép többi felhasználója számára.
- **Erőforráskorlátok.** A `--max-kv` elutasítja azokat a kéréseket, amelyeknél a KV-gyorsítótár túl nagy lenne (alapértelmezés 8g); a `--max-heavy` korlátozza az egyidejű tokenizálási, sablon- és ellenőrzési kéréseket (alapértelmezés 4); a `--keys-file` sebességkorlátai minden számítást igénylő útvonalra vonatkoznak; a darabolt törzs dekódolója lineáris; a `POST /v1/quotes/verify` plafont szab a szöveg- és forrásméretekre.
- **Naplók.** A prompt szövegét nem naplózzuk, hacsak nem adod meg a `--log-prompts` kapcsolót; a `--no-log` kikapcsolja a naplót.
- **Webes felület.** Maga az oldal nyilvános (nem tartalmaz titkot); minden hívásához token kell.
- **Megmaradó korlát:** a szerver sima HTTP-t beszél. Kösd a `127.0.0.1` címhez (ez az alapértelmezés), vagy tegyél elé TLS-t.

## Háló (mesh)

- Kölcsönös HMAC-SHA256 kézfogás legalább 8 karakteres közös kulccsal, MAC minden kereten, plafon a kézfogás és a keretek méretén.
- Az UDP-felderítés bekapcsolásra épül (`--discovery`, `--discovery-port`), a válaszok pedig nem nagyobbak a szondánál, így a munkás nem használható erősítőként.
- **Megmaradó korlát:** a forgalom hitelesített, de nem titkosított; a kulcsot a háló minden tagja ismeri; csak loopbacken tesztelve.

## Fájlok, letöltések és összefésülések

- A GGUF-elemzés megkapja a valódi fájlhosszt (`parse_with_len`); a tokenizáló felépítése trie-t használ a darabméret plafonjával; a csonka és sérült fájlok hibát adnak az adverzális tesztekben.
- Letöltések: commit-SHA-hoz kötés, méretplafonok, a sima http-re mutató átirányítások elutasítása, névtelen `wget` tartalék (a tokent sosem kapja meg), SHA-256-ellenőrzések.
- Az összefésülések nem követik a szimbolikus hivatkozásokat, és nem írnak felül kimenetet, hacsak nem adod meg az `--overwrite` kapcsolót; az alakokat és a tartományokat írás előtt ellenőrzik.
- A képdekóderek foglalás előtt érvényesítik a `Limits` korlátokat, és korlátozzák a deflate-kibontást; a WAV-olvasónak vannak ellenséges fájlokra szabott korlátai.

## Dekódolás és ellenőrzés

- A nyelvtannal, reguláris kifejezéssel és JSON-sémával történő dekódolásnak munkakeretei vannak; az idézetzár lineáris, plafonokkal, és `limit_hit` jelzést ad, ha egy plafon megállítja.
- A nyugtáknak méretplafonjaik vannak; a pillanatképek kulcsos HMAC-kal pecsételhetők; a nyugták és a pillanatképek SHA-256-láncokat használnak.
- Az átvizsgálásban talált helyességi javítások: elavult valószínűségek top-k/top-p után (most újranormalizálva), a `typ_p` álnév, tizedesvesszők a számkezelésben, előretekintési korlátok a környezeten túl, a DeepSeek-V2 csoportpontszáma, a kvantált kötegek pontossága (`exact_batch`).

## Nincs kész vagy nincs ellenőrizve

- Nincs harmadik fél általi audit, nincs a tároló adverzális tesztjein túli fuzzing-kampány, nincs a `corbul serve` penetrációs tesztje.
- A `ToolStream::push` ismételten újraolvashatja a pufferét (négyzetes a kimenet hosszában a `--max-tokens` értékig).
- A `GrammarCache` létezik, de a parancssor még nem használja.
- A függőségek licenceit utoljára a `cargo-deny` eszközzel ellenőriztük a b496ba5 commitnál.
- A CI-munkafolyamatok sosem futottak; az ARM, a macOS és a Windows kódútjait itt nem futtattuk.
