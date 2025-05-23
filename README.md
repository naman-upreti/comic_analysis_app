
---

# Comic Analysis App

An interactive data analysis and visualization web application for exploring comic book characters from Marvel and DC Universes.

This project utilizes character data scraped from Marvel Wikia and DC Wikia to uncover patterns, insights, and trends related to gender, identity, alignment, and appearances in comic books.

## Dataset Source

This project is based on the dataset used in the article:  
[Comic Books Are Still Made By Men, For Men And About Men](http://fivethirtyeight.com/features/women-in-comic-books/)

Data was scraped from:  
- [Marvel Wikia](http://marvel.wikia.com/Main_Page)  
- [DC Wikia](http://dc.wikia.com/wiki/Main_Page)

### Dataset Files:
| File Name            | Description                                |
|---------------------|--------------------------------------------|
| `dc-wikia-data.csv` | DC Comics characters data                 |
| `marvel-wikia-data.csv` | Marvel Comics characters data           |

---

## Features

- Detailed character-level analysis for Marvel and DC Universes.
- Visual exploration of:
  - Gender and Sex Distribution
  - Alignment Distribution (Good, Bad, Neutral)
  - Identity Status (Secret/Public Identity)
  - Character Survival Status (Alive/Deceased)
  - Appearance Frequency
  - First Appearance Timeline
- Comparison of Marvel vs DC characters.
- Filter and Search functionality.

---

## Dataset Columns Description

| Column Name         | Description                                         |
|--------------------|-----------------------------------------------------|
| `page_id`          | Unique page identifier within the Wikia             |
| `name`             | Character name                                      |
| `urlslug`          | Character page URL slug                             |
| `ID`               | Identity status (Secret Identity, Public Identity, etc.) |
| `ALIGN`            | Alignment status (Good, Bad, Neutral)               |
| `EYE`              | Eye color                                           |
| `HAIR`             | Hair color                                          |
| `SEX`              | Sex of the character (Male, Female, etc.)           |
| `GSM`              | Gender/Sexual Minority status                      |
| `ALIVE`            | Alive or Deceased status                            |
| `APPEARANCES`      | Number of comic book appearances (as of Sep 2014)   |
| `FIRST APPEARANCE` | Month and Year of first comic book appearance       |
| `YEAR`             | Year of first comic book appearance                 |

---

## Tech Stack Used

- Python
- Pandas
- Matplotlib / Seaborn / Plotly (for visualizations)
- Streamlit (for interactive web application)
- Jupyter Notebook (for initial data exploration)

---

## Installation

```bash
git clone https://github.com/your-username/comic_analysis_app.git
cd comic_analysis_app
pip install -r requirements.txt
```

---

## Running the App

```bash
streamlit run app.py
```
---

## Future Enhancements

- Advanced filtering options
- Character comparison dashboard
- Sentiment Analysis on character descriptions
- Network Graph of Character Relationships
- Marvel vs DC Statistical Insights Dashboard

---

## Acknowledgements

- Data Source: [FiveThirtyEight](https://fivethirtyeight.com/)
- Original Dataset Credits: Marvel Wikia & DC Wikia
- Visualization Inspiration: [FiveThirtyEight Article](http://fivethirtyeight.com/features/women-in-comic-books/)

---

