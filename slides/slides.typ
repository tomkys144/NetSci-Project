#import "@preview/definitely-not-isec-slides:1.0.1": *


#show: definitely-not-isec-theme.with(
  aspect-ratio: "16-9",
  slide-alignment: top,
  progress-bar: true,
  institute: [Network Science 2025],
  logo: [#tugraz-logo],
  config-info(
    title: [Analysis of brain blood supply and it's resistance to Thrombosis],
    subtitle: [Empirical Analysis of the Dataset],
    authors: ([Denis~Dagbert], [Tomáš~Kysela], [Hussain~Miraah~Rasheed], [Muhammad~Zubair]),
    extra: [],
    footer: [Denis~Dagbert, Tomáš~Kysela, Hussain~Miraah~Rasheed, Muhammad~Zubair],
    download-qr: "https://github.com/tomkys144/NetSci-Project/raw/refs/heads/main/slides/slides.pdf",
  ),
  config-common(
    handout: true,
  ),
  config-colors(
    primary: rgb("e4154b"),
  ),
)

#title-slide()


#slide(title: [Dataset])[
  - Whole mouse brain vasculature
  - Already preprocessed @suprosanna_2021_5367262
  - Datasets used are CD1-E-2 @Todorov2020311 as production and Synthetic graph 1 @SCHNEIDER20121397 for local development
]




#slide(title: [Bibliography])[
  #bibliography("bibliography.bib", style: "ieee")

  #text(size: 24pt, weight: "semibold")[Acknowledgments]


  #image(
    "gh.svg",
    width: 30%,
  )
  Special thanks to #link("https://gamerhost.pro/home")[Gamerhost.pro] for providing computational power.
]
