# Commit ID map (2026-11-20)

On 2026-11-20 the commit messages on `fix/evaluation-integrity` were edited. The code did not
change: every rewritten commit has exactly the same file tree as before. Editing a message gives a
commit a new ID, so runs recorded before that date store the old ID in `environment.git_commit`.
This table maps each old ID to its new one, so every recorded run can still be traced to its code.

Commits recorded by archived runs (EXP-001, EXP-002, EXP-003) are marked with *.

| Old commit | New commit | Subject |
|---|---|---|
| `d53fefa0d5f2598d4eb5549cfcbf8446010a7d52` | `8f4896a8425b114b2595018303a2bf9e80e801c8` | WIP snapshot before frontend redesign (migrated from Windows copy) |
| `e7d63bba5d1597692ca73c3a1ab30bc7b8cb869e` | `c96c6d95483464204db7e8b2efb845abbba5df4a` | frontend: monochrome design foundation, honest verdicts, simpler overview |
| `026c0ab7b984f43c0ea655966b9b3790ab996432` | `0c02eac21f2749e0adc1fc4a4f945e02bfc013ae` | frontend: rewrite Results page around plain questions and verdicts |
| `0fe1d88d5739ab394eb0447a4bdb79a80c7caf4e` | `575230310f07886e4ec31d92d4ed7eb368a0fec3` | frontend: simplify Experiments list and detail pages |
| `5add34f7822d155bdc3895f2b92a6c58a229c72c` | `68540740a14e5c47071fe061b0cc7396d754062d` | frontend: plain-language Analysis tabs; simpler comparison and failure views |
| `8094502bb0d8923a70bec81c520fcf64ac150052` | `a573503f2cf00d6dd05c7886802049583b57ce78` | frontend: fix Retrieval tab showing 0 RAG cases in live mode |
| `6d9591e679d64b8b771b5fd30775307737740bb2` | `dc79f21a39bea7cae92662bbaac972cbc4c943fb` | frontend: plain wording for Retrieval, Consistency and Speed tabs |
| `f59934ba384926afd6afdd69dc05b9431c8174d5` | `88f616ad7c86c5d07dd9d1afa6b8335b8497fecb` | frontend: review fixes for comparison, run labels, remaining wording |
| `9ab75a263bac2966fad66a4c7f6cc8d2265192c7` | `8fe23883607b4dd74ae1dde6cfa6765f6c42aa49` | Delete unused frontend code and 31 junk test experiments |
| `45bbfb5fb5bd2320c454885bd2a7b56fcab3976b` | `2c6f607c3c1b93bc5c2a87681d7cb8b1124a2501` | frontend: add README explaining concepts, usage and code structure |
| `33872b32ef7143835aacd5d30586b10af704b273` | `66435e6472601ffcb71ce6b48afe755764522c54` | api: make create_app(settings) override get_settings so routes use the given store |
| `4cfc4af2c4dc8624783729b5ddb77a730b6f952d` | `d9b3b227bbbfd7eed2f2889bca06fcfab1ee1978` | api: fix fake judge reply contract; reject retrieval metrics as answer evaluators |
| `c99995e45842aa29a05d803706684db0dcac4513` | `14a0d36ee97a98f5f29fe079b2347c4595d29c35` | runs: record executing adapter; providers send the right model; honest latency |
| `a6fd416e9a56c3b058301bbabe8afd9bf64c0e93` | `04b3cc128a937424ae55752066a97063b02e92fa` | api: reject credential-like keys in experiment definitions instead of persisting them |
| `5e4a91e85a1ed998217cfd6a938425e071565466` | `1e8ad33b2012aef3efc5f2fd62fdf0fa39aaab90` | build: declare openai and mypy; ruff and mypy clean; scripts stop writing to ./experiments |
| `eb717bf52bc772a106eb64326c458034cf1caad1` | `5bc628eddaf86b0c31153342a2fa0c4198e5e23e` | analytics: different adapters are not repeated trials; allow reference-free test cases |
| `c949beab5e1ff5dedba73dd22ffbb1e1cfac016b` | `274070eea2c2361870a26fd2f9ff3ec8701da47e` | fixtures: regenerate with the fixed judge contract and adapter provenance |
| `e45df76da1ae1dd372e29063a5ebf22a052e3dbc` | `f7e31289b2c1c53ce08cc89aa4081ec1c3250d9d` | provenance: label simulated runs from one server-side source; enforce evaluator configs |
| `68339c08230563cbd1e7762eac01159d7602134d` | `3960d092c1c5facf1c39454c19602694b1157f80` | diagnostics: attribute to the earliest failing stage; score retrieval at the run's k |
| `07047aede2f71d8fa9e80d53301d9e2c54eaa465` | `48b0047bd94dfe2f67ccdfd9d257503b11042255` | rag: refuse a pipeline whose retrieval settings differ from the experiment's |
| `060a00b341c7e2078dc74876e6fcebaca1d01b71` | `1cda8b62659cde7d7fb10c4fb3d106cc39e00216` | frontend test: non-RAG experiments default to the mock adapter |
| `ef17c85ce2b82f5b8cabfe5d863a458f12a37fa9` | `54b7d18070f90c85a12434330fbae74054c138cc` | analytics: Cohen's kappa and 2x2 table for evaluator agreement |
| `db1d699bb8c4e017b627a886badf5ee01596a42f` | `6d550351e6a9fad0567240e4eede1de636b363c9` | study prep: SQuAD subset, closed-book adapter, concise RAG prompt, SQuAD EM/F1, real embeddings |
| `a91d972cc6b667bb9939511375ae454713156774` | `4e9ae9931eb179d9f977a62a210ff645b9fe4a5b` | study: EXP-001 runner (local Ollama models, 6 arms, real judges, provenance) |
| `c9f5cdf9fdea175a769a7a09191ff5a4c5b00143` | `f2f57d641ae03a264dbddf7239a1661f3cf6e6ef` | data: reseed demo experiments and fixtures (judge config now records full judge settings) |
| `81b78ff158a9d811db69f767b77b674950f97b62` | `6cbc9ac52402299b684c44411646e97bf848d602` | study analysis script; show evaluator agreement (kappa) on the Results page |
| `4d71e75901b45caacde92eaba5cf5e984f31b11b` | `f32b48322c8bcec688f86b03166006f4ce7ee3bd` | dev: plain pytest and npm test work; EXP-001 definition; drop stray frontend store |
| `56493814bb9fcecb433e5b295ef9c7f8ddd5d111` | `9273f4d8e45e910d5c7ebe3a63d7c45ddac970d1` | docs: technical understanding report (A-K) and README with verified commands |
| `b946193da9cbdd393851e58786ea3939a0d59435` | `26295dda4a0fdad2844b1788a14a2d31db902109` | study: warm up models before each arm; export per-case CSV (annotation sheet) |
| `f6e339358bf4cf74e36e42d6fc05f3b089eca1d0` | `6cfe62865b4202f734173161c4d46bfa1981f810` | fix review findings: kappa reaches the API; legacy judge configs execute; broader credential check |
| `b3e74a2beb17dc3611256feb697e75a8feaacf99` | `291de23e2f0790e1b7511ee0227c929bdd118b30` | EXP-001 results: 540 real local-model answers, analysis, failure analysis |
| `b813f402960f762b3938da101744f15bd1fd156d` | `852986a0e5f319d7b23a30afb0a852f84688305f` | console: 'checks disagree' verdict, compare any two runs, real telemetry per question |
| `b2a28faa215c204a7c1796c6d3d71811829db09e` | `7099c255d87256d76268c8e002ae8558a267d445` | docs: final counts and console changes in report/README |
| `7e0604de2a86e5648932dbb37c0862b85473dd0d` | `9d77f2b9d84cfb8cf9a15c1013d94b3de6f4662c` | durability and guards: archive EXP-001 runs; validate benchmark adapters up front; env guard |
| `6c00f708d94f00f953fa020a6ad60fa9b3f5facf` | `bb32a54e3c541976b796d250a680f5d6d06ecf52` | docs: technical understanding and research architecture report (Markdown + PDF) |
| `ff575789336b1e635a0267084aff901290e0ff2b` | `55a0057ed653b537e6dfa36aadd2c78e4415a45a` | chore: remove placeholders and dead code; rewrite README to verified facts |
| `44c444deb26cfca586f66efc28c959d8824fe5ba` | `3c143711e586fa068536982e2a09e81e4e64cbf1` | feat: Precision@K retrieval metric; record software environment on every run |
| `1e97dab228ca0071546b204312393edc734cd56d` | `044e4e2e4a55a76ac6e092c9f42cced7948a85f0` | feat: real models from the API: SQuAD preset, live RAG, live judges, background jobs |
| `6dc586b74fcbd7761d4cefcff1e2d29ce61e496d` | `b24c5391a13154c51ed34f07089189460b7a0f97` | feat(console): run real models from the UI; SQuAD preset; live judges; job progress |
| `3c2a95731a074e3d449cc4fadb98b3d1c5937e49` | `9c8e3b0cd18e49702d280f8f7c7d7f909cffef97` | feat(rag): controlled context perturbations for RQ5 (order, distractor) |
| `166e7c1694369bc682d6a09625f9b2dae55f86e2` | `93120e2eaeabbec0e1415bc90ef36b9ec35729b1` | research(EXP-002): pre-registered robustness study definition and frozen paraphrase set |
| `129f93edb21b0f8748ecf303cee0462bdb7330e4` | `4f5bb609dd80ba681001e2a988327bab193a2afb` | research(EXP-002): runner and analysis scripts; prompt-template perturbation type |
| `b5e761eab0325f00a45a2cec7964076705a56578` | `bb7e7f9fc4666b7e6efece22cb26126b59d9d2c1` | feat(judges): rubric v2 for correctness and faithfulness; replay adapter |
| `dcedc3b9073f5baa0b9e37af4a2b8d987e16036d` | `0dc4c3e48a521a85b2682bba08cbe5840ef75045` | research(EXP-003): pre-registered judge rubric v2 re-judging study and script |
| `f6d4a6a9dc96b3f58f38cff473f4a7396207b87c` | `ad26c6fd286dd03ae71dc046f822423d8eb74c5f` | feat(console): label re-judged (replay) runs; choose judge rubric version |
| `3f6bb3256e37c8d13d7d185c1018b25641f0d3f6` | `0181fa41a8e037a40b43ef9acbff1b9392da3b33` | research(EXP-002): archive disrupted attempt 1 (Ollama OOM-killed twice) |
| `b26f573f29675342dd410e6c3025750ac20134fb` | `d8c8335c89baab8ebc74142a12d150e1df277dfc` | feat(analytics): repeated-run consistency (RQ4) as an API metric |
| `92abfc6633a57ff3e8ba4c8b45197540805290a9` | `185a41298c8bf2cbad5d0044cf6c025ce8b0b0fa` | feat(console): "Do repeated runs agree?" panel on the reliability tab |
| `295ab765889eb1a3cb40f30c3222d81a4b040d30` | `b83897ba30eeada88279a83da985dcf09402890c` | test: consistency endpoint separates adapters; panel mounted on reliability tab |
| `a92d3da0cf5c4721d2dd9e418390f73030085f55` | `6818e16ad6f9da3f32a56298ee4f211c14184e7f` | Merge branch 'rq4-consistency' |
| `739974a5c795b90075877379f9726db4f046e3ac` | `8ba561f3caec8f4f63c79959af4c0cbcac1ffa81` | refactor(analytics): move Wilson interval and exact McNemar into analytics.proportions |
| `27d7acac19ab3d4141d73cd7d0f18533f6edb409` | `75581ed2bf40441212fd6063581d19c62dc37659` | feat(annotation): tooling to validate evaluators against human correctness labels |
| `d76fa1262727d413ad2d27eaeb3d843e7609f3f7` | `c1bd669f0ecd2711d9ffec8a218187e5ba11da65` | fix(annotation): make rules 7-9 consistent with the unsure example; accept BOM CSVs |
| `03d27805e5b151b0de93696636f275e41d845e56` | `628f184b573e4a44a577e0aeb40a998b6e4d8683` | Merge branch 'human-annotation' |
| `1d698c476fa045f9269cc6a07c104d34c528822e` | `3c188317a0fbf63eae0850da7d516a0b221a83fc` | build: declare 'embeddings' extra; type-check optional ML imports via mypy overrides |
| `f5035646313454226aa33c4cf11013ec42f7d829` | `5b955259eac0a08fe6d6222f77351686504c9672` | fix(semantic): make the real BERTScore backend reproducible and safe on edge cases |
| `8a9b3762e906691c152ccea050e2075d64293753` | `5b1762625fa9dc880af18a4c470d4213e7ac9a2c` | test(rag): opt-in end-to-end SentenceTransformer retrieval on the SQuAD subset |
| `2e2b0912c8d67176247d5cb9d3803f697b1a180e` | `ba91a372288783eccaf190105a63d9e88a52f9c9` | fix(semantic): report a model with no default BERTScore layer as invalid configuration |
| `85d86ced07e15d9194067254e32daa0c1dd1641a` | `709c5e91b888236e670bdc064e624318e823b293` | Merge branch 'verify-semantic' |
| `37260d010853ec3e40d96a1b1975695cada35a89` | `e3913993888e62756800e31fd9423989df3bcbb3` | fix: review findings on merged branches (semantic, consistency) |
| `78c68af851a27a690bceb7d5d954dbc14650809e` | `810e1fa0a3100dda8cef2452d0a0e19614abd84c` | fix(annotation): keep guideline-example verdicts out of human evidence |
| `5188c4524fb8865df3c8d39bfb99fbdb55c98844` | `dd2379d2e66e04148c78165feb942f2195abf07f` | fix(annotation): rule 4/6 point to rule 10; no chunk-relative byte offset in UTF-8 error |
| `dd1aabbf4802a376b975a3c3da6a1796ec87196b` | `c2e6d80ba3dec6ab29af99053f5e96291dd8b895` | docs: README and scripts README for live models, jobs, datasets, consistency, studies |
| `80f0655ef50958bc858d2184b873faf2e322082e` | `42852a14f45952f0f6b0b820ee2a48c9ba3357d1` | research(EXP-002): results of the clean run (attempt 2) with archived runs |
