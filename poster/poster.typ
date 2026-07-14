#import "@preview/peace-of-posters:0.5.6" as pop
#import "theme.typ"
#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node

#set page("a0", margin: 1cm)
#pop.set-poster-layout(pop.layout-a0)
#pop.set-theme(theme.tug)
#set text(size: pop.layout-a0.at("body-size"))
#let box-spacing = 0.8em
#set columns(gutter: box-spacing)
#set block(spacing: box-spacing)
#pop.update-poster-layout(spacing: box-spacing)
#set math.equation(numbering: "(1)")

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
    #super("2")Institut Mines-Télécom Nord Europe, France
    #super("3")Czech Technical University in Prague,~Czechia
  ],
  logo: square(stroke: none, width: 10em)[
    #align(horizon)[
      #image("img/TU_Graz.svg", width: 100%, alt: "Logo of Graz University of Technology")
    ]
  ],
  text-relative-width: 80%,
)

#columns(2, [
  #pop.column-box(heading: "Abstract")[
    This study simulates ischemic event within high-resolution brain microvascular networks to evaluate regional vulnerability. We model blood flow using sparse matrix solvers and track the spatiotemporal progression of hypoperfusion.
  ]

  #pop.column-box(heading: "Network Topology")[
    - The dataset is sourced from VesselGraph @suprosanna_2021_5367262 and @Todorov2020311.
    - Models the complete microvascular network of a CD1-Elite mouse.
    - Most nodes exhibit a degree of 1 to 3, consistent with biological branching.

    #figure(
      image("img/graph.svg", height: 15em, alt: "3D visualization of the brain microvasculature network"),
      caption: [Spatial visualization of the dataset, with 0.5 % biggest vessels highlighted],
    )
  ]

  #pop.column-box(heading: "Flow Simulation Methods")[
    Flow through the graph is calculated using #emph(text(fill: rgb("#e4154b"))[simplified Poiseuille's Law]), shown in @eq:poiseuille.  #math.equation($frac(pi, 8 eta, style: "skewed")$, alt: "pi over 8 eta") was omitted from the full formula , since result is later normalised and calculating dynamic viscosity is beyond the scope of this project.

    #emph(text(fill: rgb("#e4154b"))[Inlets and outlets]) are identified as vessels with radius in the top 0.5 quantile. These edges are then clustered into two groups using K-means, where the more dorsal group's vessels is marked as inlets.

    #math.equation(
      $ Q = (Delta P dot r^4) / L = Delta P dot C $,
      block: true,
      numbering: "(1)",
      alt: "Poiseuille's Law",
    ) <eq:poiseuille>

    #text(size: 0.6em)[
      #math.equation($Q$, alt: "Flow Q") is flow through the edge,
      #math.equation($C = frac(r^4, L, style: "skewed")$, alt: "Capacity C") capacity of the vessel segment,
      #math.equation($Delta P$, alt: "Pressure drop") is pressure difference across the vessel segment,
      #math.equation($r$, alt: "radius")~is average radius of the segment and
      #math.equation($l$, alt: "length") is segment's length
    ]
  ]

  #pop.column-box(heading: "Simulation")[
    - #emph(text(fill: rgb("#e4154b"))[Two-stage ischemic event])

    #align(center)[
      #diagram(
        node-fill: gradient.radial(rgb("#e4154b").lighten(80%), rgb("#e4154b"), center: (30%, 20%), radius: 80%),
        node((0, 0), math.equation($r$, alt: "radius r"), radius: 1em),
        edge(`First ischemic event`, "-|>"),
        node((3.5, 0), math.equation($r/2$, alt: "radius r halved"), radius: 1em),
        edge(`Second ischemic event`, "-|>"),
        node((7, 0), math.equation($0$, alt: "zero"), radius: 1em),
      )]

    - Target is selected at random with probabilities calculated as #math.equation($P = frac(1, Q^2, style: "skewed")$, alt: "Probability P equals 1 over Q squared")

    - If Cerebral Blood Flow drops below 80% of initial flow, #emph(text(fill: rgb("#e4154b"))[Surgical anastomosis]) is simulated by adding extra edge.
  ]


  #pop.column-box(heading: "Cerebral Blood Flow evolution")[
    #figure(
      // This would be a line chart of stats_history["CBF_drop"] vs Iterations
      image("img/cbf.svg", height: 15em, alt: "Line chart showing Cerebral Blood Flow reduction over time"),
      caption: [CBF reduction and over iterations],
    )
    - #emph(text(fill: rgb("#e4154b"))[Primary Phase]): Slower decay as the network utilizes collateral pathways to maintain flow in peripheral nodes.
    - #emph(text(fill: rgb("#e4154b"))[Critical Failure]): Compensation is not possible and CBF quickly drops.
  ]

  #pop.column-box(heading: "Spatiotemporal Vulnerability")[
    #figure(
      image("img/hypo-time-CD1_E_no2.svg", height: 15em, alt: "Heatmap of spatiotemporal vulnerability in the brain"),
      caption: [Iterations required for a region to reach 40% hypoperfused vessels],
    )

    Based on simulation #emph(text(fill: rgb("#e4154b"))[Somatosensory Cortex and Striatum]) are first affected by ischemia. This leads to a reduced ability to sense environment using whiskers and loss of motor control.
  ]

  #pop.column-box(heading: "Results & Discussion")[
    1. #emph(text(fill: rgb("#e4154b"))[Network Resilience and Critical Failure]): Our simulation successfully identified the Somatosensory Cortex and Striatum as regions most susceptible to ischemia.
    2. #emph(text(fill: rgb("#e4154b"))[Network Resilience]): The CBF evolution profile reveals a non-linear response to vessel occlusion, with compensation phase followed by critical failure.
    3. #emph(text(fill: rgb("#e4154b"))[Efficacy of Anastomosis]): Simulation shows a delay of critical failure, but it is not significant without further treatment.
    4. #emph(text(fill: rgb("#e4154b"))[Limitations & Future Work]): The model does not take non-newtonian behavior of blood and dilation/constriction of vessels into account. This could be fixed for better accuracy.
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
      *Computational resources:* #linebreak()
      #box(height: auto)[#image("img/gh.svg", height: 2em, alt: "GamerHost Logo")] #h(1em)
      #box(height: auto)[#image("img/metacentrum.svg", height: 2em, alt: "Metacentrum Logo")]
    ],
    align(top + left)[
      #v(0.8cm)
      #set text(size: 1em) // Small font for bibliography to save space
      *Bibliography:* #linebreak()
      #v(.4em)
      #set text(size: 0.4em)
      #show bibliography: set block(spacing: 0.1em)
      #bibliography("bibliography.bib", style: "ieee", title: none)
    ],
  )
]
