import pandas as pd

# Datei laden
file_path = 'Datafiles/2_bundesliga_spielplan_corrected_semicolon.csv'  # Pfad zur Datei
data = pd.read_csv(file_path, sep=';', error_bad_lines=False)  # Fehlerhafte Zeilen überspringen

# Annahme: Die Spalte mit dem Datum heißt 'Datum'. Passe den Namen an, falls nötig.
data['Date'] = pd.to_datetime(data['Date'], format='%d.%m.%Y', errors='coerce').dt.strftime('%d/%m/%Y')

# Geänderte Datei speichern
output_path = '2_bundesliga_spielplan_converted.csv'
data.to_csv(output_path, sep=';', index=False)

print(f"Die Datei wurde erfolgreich konvertiert und gespeichert unter: {output_path}")
