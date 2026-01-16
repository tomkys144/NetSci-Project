#import "@preview/peace-of-posters:0.5.6" as pop
#import "theme.typ"

#set page("a0", margin: 1cm)
#pop.set-poster-layout(pop.layout-a0)
#pop.set-theme(theme.tug)
#set text(size: pop.layout-a0.at("body-size"))
#let box-spacing = 0.8em
#set columns(gutter: box-spacing)
#set block(spacing: box-spacing)
#pop.update-poster-layout(spacing: box-spacing)

#pop.title-box(
  "Mapping Ischemic Risk in Large-Scale Brain Microvasculature",
  subtitle: "Network Science 2025",
  authors: [
    Denis~Dagbert#super("1,2"),
    Tomáš~Kysela#super("1,3"),
    Hussain~Miraah~Rasheed#super("1")
    and Muhammad~Zubair#super("1")
  ],
  institutes: [
    #set text(fill: black, weight: "regular", size: .75em)
    #super("1")Graz University of Technology,~Austria
    #super("2")?, France
    #super("3")Czech Technical University in Prague,~Czechia
  ],
  logo: square(stroke: none, width: 10em)[
    #align(horizon)[
      #image("img/TU_Graz.svg", width: 100%)
    ]
  ],
  text-relative-width: 80%,
)

#columns(2, [
  #pop.column-box(heading: "Abstract")[
    This study simulates thrombosis within high-resolution brain microvascular networks to evaluate regional vulnerability[cite: 12]. We model blood flow using sparse matrix solvers and track the spatiotemporal progression of hypoperfusion.
  ]

  #pop.column-box(heading: "Network Topology")[
    The dataset (CD1-E_no2) consists of a complex graph where nodes represent vessel junctions and edges represent vessel segments.
    #figure(
      image("img/sbm-CD1-E_no2_radial.png", height: 15em),
      caption: [Radial Stochastic Block Model (SBM) showing the community structure of the brain network.],
    )
    - *Degree Distribution:* Most nodes exhibit a degree of 1–3, consistent with biological branching.
    - *Connectivity:* Inlets and outlets are identified via K-Means clustering on ventral and dorsal coordinates.
  ]

  #pop.column-box(heading: "Flow Simulation Methods")[
    where conductance is calculated via Poiseuille's law[cite: 41, 42].
    #table(
      columns: (auto, 1fr),
      inset: 0.4cm,
      stroke: (y: 0.2pt + black),
      [Inlet Pressure ($P_{i n}$)], [100.0],
      [Outlet Pressure ($P_{o u t}$)], [0.0],
      [Critical Threshold], [40% Flow Reduction],
    )
  ]

  #pop.column-box(heading: "CBF & Hypoperfusion Evolution")[
    #figure(
      // This would be a line chart of stats_history["CBF_drop"] vs Iterations
      image("img/placeholder.jpg", height: 15em),
      caption: [Global CBF reduction and hypoperfused vessel fraction over simulation time.],
    )
    - **Primary Phase:** Rapid CBF drop corresponds to the occlusion of high-PageRank "hub" vessels.
    - **Secondary Phase:** Slower decay as the network utilizes collateral pathways (high clustering) to maintain flow in peripheral nodes.
    - *Critical Failure:* The point where the 40% hypoperfusion threshold is reached in the "centerpiece" slices.
  ]

  #pop.column-box(heading: "Spatiotemporal Vulnerability")[
    #figure(
      image("img/placeholder.jpg", height: 15em),
      caption: [3D Sliced Map: Color represents the simulation steps required for a 'pixel' to reach the 40% hypoperfusion threshold (Ischemic Penumbra).],
    )
    This visualization tracks the evolution of ischemic clusters over simulation time.
  ]

  #pop.column-box(heading: "Results & Discussion")[
    - *Global CBF Drop:* Tracking total flow reduction as thrombosis propagates.
    - *PageRank Centrality:* Higher centrality nodes correlate with faster regional failure.
    - *Redundancy:* High local clustering provides alternative pathways, delaying the onset of critical hypoperfusion.
  ]
])

#pop.bottom-box(
  heading-box-args: (
    fill: none,
    stroke: (
      top: .1em + rgb("#e4154b"),
    ),
    outset: (top: .1em),
  ),
  heading-text-args: (
    fill: rgb("#e4154b"),
  ),
)[
  #grid(
    columns: (2fr, 2fr),
    // Left for logos/resources, Right for bibliography
    align(top + left)[
      #v(0.8cm)
      #set text(size: 1em)
      *Computational resources:* #linebreak() #v(.2em)
      #box(height: 1.5em)[#image("img/gh.svg", height: 2em)] #h(1em)
      #box(height: 1.5em)[#image("img/metacentrum.svg", height: 2em)]
    ],
    align(top + left)[
      #v(0.8cm)
      #set text(size: 1em) // Small font for bibliography to save space
      *Bibliography:* #linebreak() #v(.4em)
      #set text(size: 0.35em)
      #show bibliography: set block(spacing: 0.1em)
      #bibliography("bibliography.bib", full: true, style: "ieee", title: none)
    ],
  )
]
