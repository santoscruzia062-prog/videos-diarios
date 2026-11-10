# Histórico de vídeos

Uma linha por vídeo publicado/agendado (mais novo em cima). Use para não repetir formato+tema
dos últimos 7 dias.

| Data | Horário | Formato | Tema | Formato da arena | Resultado |
|---|---|---|---|---|---|
| 2026-11-10 | 21:00 | refeito de manhã (ref. pasta do dono) — pieces minimal | 3 braços + 2 prensas (pentagono_rotor) | pentágono | agendado |
| 2026-11-10 | 18:30 | refeito — pieces minimal, manchas | losango aberto girando (diamante_aberto) | círculo | agendado |
| 2026-11-10 | 16:00 | refeito — pieces minimal, manchas | 4 arcos entrando pelos cantos (quadrado_arcos) | quadrado | agendado |
| 2026-11-10 | 13:30 | refeito — pieces minimal | colchetes no trilho (colchetes_trilho) | círculo | agendado |
| 2026-11-10 | 11:30 | refeito — pieces minimal, accel + RPM | barra que acelera 10→54 RPM (acelera_martelo) | círculo | agendado |
| 2026-11-10 | 10:00 | refeito — pieces minimal | X no meio + setas dos lados (octogono_x_setas) | octógono | agendado |
| 2026-11-10 | 08:30 | bolinhas multiplicando (bolas.py, pieces/prensas, minimal azul-marinho) | 4 prensas metálicas (prensas_4) | círculo | agendado |
| 2026-11-10 | 07:00 | bolinhas multiplicando (bolas.py, rods, minimal) | roda de 5 raios (minimal_roda_raios) | círculo | agendado |
| 2026-11-09 | 21:00 | bolinhas multiplicando (bolas.py, rods) | 3 bastões girando | hexágono | agendado |
| 2026-11-09 | 18:30 | bolinhas multiplicando (bolas.py, rods) | 4 bastões | círculo | agendado |
| 2026-11-09 | 16:00 | bolinhas multiplicando (bolas.py, gears) | engrenagens | triângulo | agendado |
| 2026-11-09 | 13:30 | bolinhas multiplicando (bolas.py, rods half+marks) | relógio com ponteiro | círculo | agendado |
| 2026-11-09 | 11:30 | bolinhas multiplicando (bolas.py, rods) | 2 bastões | coração | agendado |
| 2026-11-09 | 10:00 | quebra-quadrados (quebra.py, k=2.65) | 1 bolinha vs 108 quadrados | retângulo arredondado | agendado |
| 2026-11-09 | 08:30 | bolinhas multiplicando (bolas.py, rods) | 4 bastões | quadrado | agendado |
| 2026-11-09 | 07:00 | bolinhas multiplicando (bolas.py, bounce) | bolinha quicando, cada batida +1 | círculo | agendado |
| 2026-11-08 | 20h | duelo (arena.py) | 🦈 TUBARÃO vs 🐊 JACARÉ (tubarão 339 x 312) | pentágono | agendado |
| 2026-11-08 | 16h | batalha royale (royale_times.py) | 8 times brasileiros (Vasco campeão) | círculo | agendado |
| 2026-11-08 | 13h | batalha royale (royale.py) | 8 países (BR campeão) | círculo | agendado |
| 2026-11-08 | 11h26 | duelo | 🦁 LEÃO vs 🐺 LOBO | octógono | publicado |
| 2026-11-07 | — | duelos feitos (alguns postados manualmente pelo dono) | ☀️ SOL vs 🌚 LUA, 😍 AMOR vs 👿 ÓDIO, 🐶 CACHORRO vs 🐱 GATO, ⚽ FUTEBOL vs 🏀 BASQUETE, 💎 DIAMANTE vs 🪙 OURO, 🍕 PIZZA vs 🍔 HAMBÚRGUER | estrela, coração, quadrado, círculo, losango, retângulo | — |
| 2026-11-07 | — | duelos antigos | 😇 BEM vs 😈 MAL; 🔥 FOGO vs 🧊 GELO | hexágono; triângulo | — |
| 2026-11-07 | — | multiplicação | 🤑 dinheiro; ❤️ corações; bolinhas coloridas | quadrado; círculo | — |

## Referências do dono (PRIORIDADE — usar 2–3 por noite, misturando com o nosso estilo)
Ref. 3 (10/11, 10 vídeos na pasta do PC do dono "EDITOR DE VIDEO IA/REFERENCIAS"): fundo preto puro,
arena com linha branca fina, peças cinza que SAEM para fora da arena (trilhos, braços, arcos),
cores que "tomam conta" em ondas, final com a arena sumindo e as bolinhas voando. Ideias ainda não
feitas: corrida de 4 cores "Which color clears the arena first?" quebrando blocos; anéis
concêntricos de bolinhas sendo "contaminados" por uma bolinha branca (vídeo curto, 15 s);
octógono com zonas x5/x20/BIG/SMALL que mudam a bolinha; "Every smash speeds up" com RPM subindo.
Evitar: "#" que cruza (cria compartimentos vazios) e raios longos lentos (deixam buracos).
**✅ FEITO em 09/11 (vídeo de 10/11 10:00, `motor/duelo_itens.py`): 🇺🇸 USA vs BRAZIL 🇧🇷 — duelo com itens.** Dá para reaproveitar o motor para outros pares de bandeiras/animais (editar `CONFIGS`/`script`).
Código novo (ex.: `motor/duelo_itens.py`, pode reaproveitar partes de `arena.py`/`bolas.py`):
- Círculo com linha branca fina, fundo azul-marinho escuro com grade sutil.
- 2 bolões-personagem com a BANDEIRA (emoji 🇺🇸 / 🇧🇷 da Noto Color Emoji) e anel colorido
  (azul/vermelho x verde/amarelo) quicando pela arena.
- Topo: "USA vs BRAZIL" + 2 BARRAS DE VIDA com a bandeira em cada ponta.
- A cada ~4 s aparece um ITEM num anel brilhante (espinho, espada de brinquedo, escudo, ímã,
  raio). Quem encosta pega → aviso grande "BRAZIL GOT THE SPIKE!" / "USA GOT THE SHIELD!".
  Com item, o choque tira vida do outro; dano = faíscas/confete na cor do time (NADA de sangue,
  NADA de arma de fogo realista).
- Quem acerta solta bolinhas da sua cor (multiplica) → arena vai lotando nas duas cores.
- Vida bem disputada, vira no final; quem zera perde → explosão de confete na cor do vencedor,
  "BRAZIL WINS!" / "USA WINS!", "Who were you rooting for? 👇", "Follow for more!".
- Tudo em inglês. Legenda: "🇺🇸 USA vs BRAZIL 🇧🇷 Who wins this battle? …" #usavsbrazil #battle #satisfying
- NUNCA usar pessoas reais / políticos / fotos de gente (o dono pediu Lula/Trump/Flávio e eu recusei:
  risco de denúncia e remoção). Só bandeiras, animais, elementos ou cores.

Ref. 2 (canal de física "satisfying", 08/11): fundo azul-marinho bem escuro, círculo com linha
branca fina, peças METÁLICAS cinza com sombra (visual de máquina/laboratório), bolinhas PEQUENAS
(raio ~4–5) e poucas no começo, contador = só um NÚMERO grande no topo (sem texto).
- [x] (10/11, prensas_4) 4 PRENSAS/pistões metálicos presos na borda (cima, baixo, esquerda, direita) que descem e
      batem; bolinha esmagada multiplica (o mais visto da referência, 9,6 mi). Movimento ritmado
      de prensa (é outro mecanismo; o "nada de vai-e-vem" vale para os giratórios).
- [ ] 2 prensas largas horizontais (cima e baixo) fechando como um sanduíche.
- [x] (10/11, rotor_cunhas) Rotor com 2 pás em CUNHA largas (tipo hélice/leque) girando no centro.
- [ ] 6–7 cunhas metálicas fixas na borda apontando para dentro + bolinhas quicando nelas.
- [ ] 2 hastes diagonais com placa na ponta (formato T) girando em sentidos opostos.
- [ ] Bolinha-emoji ❤️ no centro + texto "Manda esse vídeo pra quem você ama ❤️" (gera envio).
- [ ] (baixa prioridade) corrida 1 vs 1 vermelho x azul "Quem chega a 5.000 primeiro?".
Ref. 1 (outro canal, 08/11): fundo preto, manchas de cor, roda de raios, moldura retangular
girando, quadrado vazado girando no centro, semicírculos nas laterais, portais x5/x20.
  Usados em 10/11: roda de raios ✅, moldura retangular ✅ (com aberturas — fechada deixava buracos fora), quadrado vazado ✅. Faltam: semicírculos laterais, portais x5/x20.
  Motor novo: `mech="pieces"` em bolas.py (peças-polígono que giram e/ou batem como prensa) — use para hastes em T, cunhas fixas, semicírculos etc.

## Ideias na fila (ainda não feitas)
- quebra_circulo (já tem config em quebra.py: quebra-quadrados dentro de círculo)
- estrela com 5 bastões curtos (formato "star" já existe; também "pentagon" e "diamond"); losango com engrenagens; pentágono com ponteiro duplo
- bolinha quicando dentro de quadrado/triângulo; 2 ponteiros (horas e minutos) no relógio
- bastões de tamanhos/velocidades diferentes; engrenagens de 3 dentes
- Batalha royale dos 12 signos (♈ — fazer distintivos coloridos, os emojis de signo são todos roxos)
- Qual mês de aniversário sobrevive?
- "A bolinha escapa?" (anéis girando com abertura; versão com 12 bolinhas de signos)
- "Vai transbordar em 30 s?" com relógio
- Pedra, Papel e Tesoura (conversão por choque)
- "Ache o emoji diferente" por níveis (participação: "até que nível você chegou?")
- "Quantas bolinhas tem?" (palpite nos comentários, resposta no fim)
- Corrida de bolinhas dos países (pista/Plinko com placar)
- Ilusões de ótica sem flashes
