# Artificial Intellgience in Industry - project

The aim of the project is to estimate air pollution according to the traffic and weather conditions. To do so, data from Bologna, Italy have been used.

## Datasets

### Air pollution

* [Dataset](https://opendata.comune.bologna.it/explore/dataset/dati-centraline-bologna-storico/table/?sort=data_inizio&disjunctive.agente)

Readings from three stations in Bologna. Each station collects a bunch of measurements, in particular:

* **NO2 (NITROGEN DIOXIDE)**
* **O3 (OZONE)**
* **CO (CARBON MONOXIDE)**
* **NOX (NITROGEN OXIDES)**
* **NO (NITRIC OXIDE)**
* **C6H6 (BENZENE)**
* **PM10**
* **PM2.5**

Some measures might be missing. For example *PM10* and *PM2.5* are not often present.

### Traffic

* [Coil Traffic readings](https://opendata.comune.bologna.it/explore/dataset/rilevazione-flusso-veicoli-tramite-spire-anno-2024/table/?disjunctive.codice_spira&disjunctive.tipologia&disjunctive.nome_via&disjunctive.stato&sort=data)
* [Coil accuracy](https://opendata.comune.bologna.it/explore/dataset/accuratezza-spire-anno-2024/information/?disjunctive.codice_spira_2)

Traffic data is measured used some specific coild that can detect if a veichle is passing over it. There are many gates around the city that collects this information. The dataset page suggests to also consider the coil accuracy dataset, used to check if the readings are correct.

### Weather

* [Temperature and precipitations](https://dati.arpae.it/dataset/erg5-interpolazione-su-griglia-di-dati-meteo)

The dataset containd the following informations:

* **TAVG:** Average Temperature
* **PREC:** Precipitations
* **RHAVG:**
* **RAD:**
* **W_SCAL_INT:**
* **W_VEC_DIR:**
* **W_VEC_INT:**
* **LEAFW:**
* **ET0:**

### Direct Downloads:

* [Air pollution](https://opendata.comune.bologna.it/api/explore/v2.1/catalog/datasets/dati-centraline-bologna-storico/exports/csv?lang=it&qv1=(data_inizio%3A%5B2018-12-31T23%3A00%3A00Z%20TO%202024-12-31T22%3A59%3A59Z%5D)&timezone=Europe%2FRome&use_labels=true&delimiter=%3B) 2019-2024 (47 MB)
* **Traffic:**

  * **2019**: [coil data](https://opendata.comune.bologna.it/api/explore/v2.1/catalog/datasets/rilevazione-autoveicoli-tramite-spire-anno-2019/exports/csv?lang=it&timezone=Europe%2FRome&use_labels=true&delimiter=%3B) / [coil accuracy](https://opendata.comune.bologna.it/api/explore/v2.1/catalog/datasets/accuratezza-spire-anno-2019/exports/csv?lang=it&timezone=Europe%2FRome&use_labels=true&delimiter=%3B)
  * **2020**: [coil data](https://opendata.comune.bologna.it/api/explore/v2.1/catalog/datasets/rilevazione-autoveicoli-tramite-spire-anno-2020/exports/csv?lang=it&timezone=Europe%2FRome&use_labels=true&delimiter=%3B) / [coil accuracy](https://opendata.comune.bologna.it/api/explore/v2.1/catalog/datasets/accuratezza-spire-anno-2020/exports/csv?lang=it&timezone=Europe%2FRome&use_labels=true&delimiter=%3B)
  * **2021**: [coil data](https://opendata.comune.bologna.it/api/explore/v2.1/catalog/datasets/rilevazione-autoveicoli-tramite-spire-anno-2021/exports/csv?lang=it&timezone=Europe%2FRome&use_labels=true&delimiter=%3B) / [coil accuracy](https://opendata.comune.bologna.it/api/explore/v2.1/catalog/datasets/accuratezza-spire-anno-2021/exports/csv?lang=it&timezone=Europe%2FRome&use_labels=true&delimiter=%3B)
  * **2022**: [coil data](https://opendata.comune.bologna.it/api/explore/v2.1/catalog/datasets/rilevazione-flusso-veicoli-tramite-spire-anno-2022/exports/csv?lang=it&timezone=Europe%2FRome&use_labels=true&delimiter=%3B) / [coil accuracy](https://opendata.comune.bologna.it/api/explore/v2.1/catalog/datasets/accuratezza-spire-anno-2022/exports/csv?lang=it&timezone=Europe%2FRome&use_labels=true&delimiter=%3B)
  * **2023**: [coil data](https://opendata.comune.bologna.it/api/explore/v2.1/catalog/datasets/rilevazione-flusso-veicoli-tramite-spire-anno-2023/exports/csv?lang=it&timezone=Europe%2FRome&use_labels=true&delimiter=%3B) / [coil accuracy](https://opendata.comune.bologna.it/api/explore/v2.1/catalog/datasets/accuratezza-spire-anno-2023/exports/csv?lang=it&timezone=Europe%2FRome&use_labels=true&delimiter=%3B)
  * **2024**: [coil data](https://opendata.comune.bologna.it/api/explore/v2.1/catalog/datasets/rilevazione-flusso-veicoli-tramite-spire-anno-2024/exports/csv?lang=it&timezone=Europe%2FRome&use_labels=true&delimiter=%3B) / [coil accuracy](https://opendata.comune.bologna.it/api/explore/v2.1/catalog/datasets/accuratezza-spire-anno-2024/exports/csv?lang=it&timezone=Europe%2FRome&use_labels=true&delimiter=%3B)
* **Weather** (`zip` files. We will only use the `01421_year_h.csv` file, containing hourly data)**:**

  * [2019](https://dati-simc.arpae.it/opendata/erg5v2/timeseries/01421/01421_2019.zip)
  * [2020](https://dati-simc.arpae.it/opendata/erg5v2/timeseries/01421/01421_2020.zip)
  * [2021](https://dati-simc.arpae.it/opendata/erg5v2/timeseries/01421/01421_2021.zip)
  * [2022](https://dati-simc.arpae.it/opendata/erg5v2/timeseries/01421/01421_2022.zip)
  * [2023](https://dati-simc.arpae.it/opendata/erg5v2/timeseries/01421/01421_2023.zip)
  * [2024](https://dati-simc.arpae.it/opendata/erg5v2/timeseries/01421/01421_2024.zip)
