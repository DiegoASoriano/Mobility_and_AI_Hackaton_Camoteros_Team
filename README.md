# Camoteros Mobility Hackathon Project: Parallel Emergency Dispatch AI

## Project Goal
Our solution tackles the critical 30-minute delay between a technical fault being detected and the manual dispatch order being formalized (C5 bureaucracy). We aim to use AI and data to dispatch the most efficient technician in parallel with the administrative process, ensuring the technician arrives at the fault location instantly when the formal order is released.

## The Solution: A Race Against Time
We use a Parallel Dispatch Strategy to convert the 30-minute bureaucratic delay into a technical travel window.

* **Problem:** The 30-minute lag between fault detection and formal dispatch order (C5).
* **Our Value:** Our system uses location data and optimization to send the closest, qualified technician immediately, using the 30-minute administrative window for travel.

---

## Technology Stack
* **Language:** Python (for all core scripts)
* **Data Processing:** Pandas, NumPy
* **AI/Optimization:** Scikit-learn (Simple Regression/Optimization for assignment)
* **Demo & MVP:** Streamlit
* **Version Control:** Git & GitHub

---

## Project Structure
To keep the project organized and professional, we use the following structure:
* **src:** Contains the main application scripts (app.py, app_technician.py) and data generators.
* **data:** Stores the generated datasets (.csv). Note: The apps read data from here automatically.
* **models:** Contains the core AI logic and training algorithms (ai_logic_core.py).

## How to Run the Demo

To run the final solution, the Platform Engineer must execute the main application script:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/DiegoASoriano/Camoteros_hackaton.git](https://github.com/DiegoASoriano/Camoteros_hackaton.git)
    ```
2.  **Set up Environment:** (Ensure you have Python 3.x installed.)
    ```bash
    pip install -r requirements.txt
    ```

3.  **Generate Synthetic Data:**
    Before running the apps, ensure the database is populated. Run these two scripts to generate 10 years of fault history and the technician roster in the data/ folder:
    ```bash
    python src/FaultHistoryGenerator.py
    python src/TechnicianInventoryGenerator.py
    ```
3.  **Run the Apps:**
    ```bash
    streamlit run src/app.py
    streamlit run src/app_technician.py
    ```
    (Note: Paths may need to be adjusted based on the final file structure.)
