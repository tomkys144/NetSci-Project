#import "@preview/peace-of-posters:0.5.6" as pop
#import "theme.typ"

#set page("a0", margin: 1cm)
#pop.set-poster-layout(pop.layout-a0)
#pop.set-theme(theme.tug)
#set text(size: pop.layout-a0.at("body-size"))
#let box-spacing = 1.2em
#set columns(gutter: box-spacing)
#set block(spacing: box-spacing)
#pop.update-poster-layout(spacing: box-spacing)

#pop.title-box(
  "Analysis of brain blood supply and it's resistance to Thrombosis",
  subtitle: "Network Science 2025",
  authors: [
    Denis~Dagbert#super("1,2"),
    Tomáš~Kysela#super("1,3"),
    Hussain~Miraah~Rasheed#super("1")
    and Muhammad~Zubair#super("1")
  ],
  institutes: [
    #set text(fill: black, weight: "regular", size: .75em)
    #super("1")Graz University of Technology, Austria
    #super("2")?, France
    #super("3")Czech Technical University in Prague,~Czechia
  ],
  logo: square(stroke: none)[
    #set align(horizon)
    #image("img/TU_Graz.svg", width: 100%)
  ],
  text-relative-width: 75%,
)

#columns(2, [
  #pop.column-box(heading: "Columbidae")[
    'Columbidae is a bird family consisting of doves and pigeons.
    It is the only family in the order Columbiformes.'

    #figure(caption: [
      Pink-necked green pigeon.
    ])[
    ]
  ]

  #pop.column-box(
    heading: "Biological Information",
  )[
    #table(
      columns: (auto, 1fr),
      inset: 0.5cm,
      stroke: (x, y) => if y >= 0 { (bottom: 0.2pt + black) },
      [Domain], [Eukaryota],
      [Kingdom], [Animalia],
      [Phylum], [Chordata],
      [Class], [Aves],
      [Clade], [Columbimorphae],
      [Order], [Columbiformes],
      [Family], [Columbidae],
      [Type genus], [Columba],
    )

    This box is styled differently compared to the others.
    To make such changes persistent across the whole poster, we can use these functions:
    ```typst
    #pop.update-poster-layout(...)
    #pop.update-theme()
    ```
  ]

  #pop.column-box(heading: "Peace of Posters Documentation")[
    You can find more information on the documentation site under
    #text(fill: red)[
      #link("https://jonaspleyer.github.io/peace-of-posters/")[
        jonaspleyer.github.io/peace-of-posters/
      ]
    ].

    #figure(caption: [
      The poster from the thumbnail can be viewed at the documentation website as well.
    ])[
      #link("https://jonaspleyer.github.io/peace-of-posters/")[
      ]
    ]
  ]

  #colbreak()

  #pop.column-box(heading: "General Relativity")[
    Einstein's brilliant theory of general relativity.
    $ G_(mu nu) + Lambda g_(mu nu) = kappa T_(mu nu) $
    However, they have nothing to do with doves.
  ]

  #pop.column-box(heading: "Peace be with you")[
    #figure(caption: [
      'Doves [...] are used in many settings as symbols of peace, freedom or love.
      Doves appear in the symbolism of Judaism, Christianity, Islam and paganism, and of both
      military and pacifist groups.'
    ])[
    ]
  ]

  #pop.column-box(heading: "Etymology")[
    Pigeon is a French word that derives from the Latin pīpiō, for a 'peeping' chick,
    while dove is an ultimately Germanic word, possibly referring to the bird's diving flight.
    The English dialectal word culver appears to derive from Latin columba
    A group of doves is called a "dule", taken from the French word deuil ('mourning').
  ]

  #pop.column-box()[
    #bibliography("bibliography.bib", full: true, style: "ieee")
  ]
])

#pop.bottom-box(
  heading-box-args: (
    fill: white,
    stroke: (
      top: .1em + rgb("#e4154b"),
    ),
    outset: (top: .2em),
  ),
  heading-text-args: (
    fill: rgb("#e4154b"),
  ),
)[
  #align(center)[
    #align(horizon)[
      Computational resources provided by #linebreak()
      #box(inset: (right: 1em, left: 1em, rest: .2em))[
        #image(
          "img/gh.svg",
          height: 2em,
        )
      ]
      #box(inset: (right: 1em, left: 1em, rest: .2em))[
        #image(
          "img/metacentrum.svg",
          height: 2em,
        )
      ]
    ]
  ]
]
