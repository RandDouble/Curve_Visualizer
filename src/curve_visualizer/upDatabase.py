import argparse
import sqlite3
import os
import glob
from datetime import datetime
from io import TextIOWrapper
from pathlib import Path

import numpy as np

# @TODO writing doc-string for all function


def check_header_line(s_descr: str, index: int, size: int) -> list[float]:
    data = s_descr.strip().split(sep="\t")
    if len(data) == 8:
        # Essentially I want to enforce a common  structure to the header
        # If this structure is present, do nothing
        return data

    b_misura = 1.0 if ((index + 1) == size) else 0.0
    f_rip = 1.0
    data.extend((f"{b_misura:.6f}", f"{f_rip:.6f}"))

    return data


def create_db(db_name: Path) -> sqlite3.Connection:
    while True:
        user_input = input(
            "Database Not Found, do you want to create it? ([yes] / no)\n"
        )

        if user_input.lower() in ["yes", "y", ""]:
            con = sqlite3.Connection(db_name)
            break
        elif user_input.lower() in ["no", "n"]:
            print("Database Not Created")
            exit(0)
        else:
            user_input = input(
                "Database Not Found, do you want to create it? ([yes] / no)\n"
            )

    cur = con.cursor()

    cur.execute(
        """
        CREATE TABLE campioni
            (
                rowid integer primary key autoincrement,
                campione text,
                date date,
                misura int,
                tipologia text,
                gen_rep int
            )
    """
    )

    cur.execute(
        """
        CREATE TABLE impulsi
            (
                rowid integer,
                periodoTot float,
                frequency float,
                activeTime float,
                activeFreq float,
                voltage float,
                dutyCicle float,
                read int,
                rip int,
                FOREIGN KEY ( rowid )
                REFERENCES campioni (rowid) ON DELETE CASCADE
            )
    """
    )

    cur.execute(
        """
        CREATE TABLE misure
            (
                rowid integer,
                times float,
                voltage float,
                current float,
                FOREIGN KEY ( rowid )
                REFERENCES campioni (rowid) ON DELETE CASCADE
            )
    """
    )

    con.commit()

    return con


def reset_db(con: sqlite3.Connection, cur: sqlite3.Cursor):
    cur.execute("update sqlite_sequence set seq = 1 where name = 'campioni';")

    con.commit()

    cur.execute("DELETE FROM campioni")
    cur.execute("DELETE FROM impulsi")
    cur.execute("DELETE FROM misure")

    con.commit()


def header_in_table(
    con: sqlite3.Connection,
    cur: sqlite3.Cursor,
    row_id: int,
    n_row_impulse: int,
    read_file: TextIOWrapper,
) -> list[float]:
    pulse = []
    for index in range(n_row_impulse):
        line = check_header_line(read_file.readline(), index, n_row_impulse)
        row = ",".join(line)
        f_pulse_value: float = float(line[4])
        if np.abs(f_pulse_value) > 1:
            pulse.append(f_pulse_value)
        cur.execute(
            "INSERT INTO impulsi (rowid, periodoTot, frequency, activeTime,"
            + f" activeFreq, voltage, dutyCicle, read, rip) VALUES ({row_id}, {row})"
        )
        con.commit()
    return pulse


def measure_in_table(
    con: sqlite3.Connection,
    cur: sqlite3.Cursor,
    row_id: int,
    read_file: TextIOWrapper,
) -> list[float]:
    check = []
    first = True  # Boolean to check if first line, to get starting time.
    start_time : float = 0
    for line in read_file:
        if len(line) < 4:  # Elimina tutto quello che ha meno di 4 caratteri
            # Skip if it has less than 4 chars.
            continue

        values = np.array(line.strip().split(sep="\t"), dtype=np.float64)
        if len(values) != 3:  # Elimina le righe vuote
            # Skip if row is incomplete.
            continue

        if first:
            start_time = values[0]
            first = False
        values[0] -= start_time
        # Resetting time to start from zero

        row = ",".join(values.astype(str).tolist())

        # print(row_id, row)

        check.append(values[1])

        cur.execute(
            f"INSERT INTO misure (rowid, times, voltage, current) VALUES ({row_id}, {row})"
        )

    if len(check) < 2: # Se ho una solo linea di misure, allora la misura è fallita.
        check = 0
        con.rollback() # Annullo la transazione con il database.
    return check


def check_type(b_pulse: bool, pulse: list[float], check: list[float]) -> str:
    # Declaring measure Type
    if b_pulse:
        if all(i > 1 for i in pulse) or all(i < 1 for i in pulse):
            tipologia = "IMPULSO_UNIPOLARE"
        else:
            tipologia = "IMPULSO_ALTERNATO"

    elif len(set(check)) == 1:
        # Controlling if check contains only one unique element
        # Necessary because new IV Curve contains more than one element.
        tipologia = "LETTURA"
    else:
        tipologia = "IV_CURVE"
    return tipologia


def clean_and_upload(
    con: sqlite3.Connection,
    cur: sqlite3.Cursor,
    search_dirs: list[Path],
) -> int:
    count : int = 0
    for main_dir in search_dirs:
        CAMPIONE = main_dir.parts[-1]
        # after main dir there are date dirs
        for time in main_dir.iterdir():
            # and now measure dir
            TIME = time.parts[-1]
            TIME = datetime.strptime(f"{TIME[0:4]}-{TIME[4:7]}-{TIME[7:]}", "%Y-%b-%d")

            for run in time.iterdir():
                if not run.is_dir():
                    continue
                    # Se non è una directory non si può fare nulla passa al prossimo.
                    # If it isn't a directory, skip it.

                RUN = run.parts[-1]

                res = cur.execute(
                    f"SELECT 1 FROM campioni WHERE campione = '{CAMPIONE}'"
                    + f" AND date = '{TIME.date()}' AND misura = '{RUN}'"
                )

                if res.fetchone() is not None:
                    # Se trova il file, vuol dire che è presente nel database e passa oltre.
                    # If file is found, it is already inside in database, so skip.
                    continue

                # Cerca il primo file di tipo txt
                # Se non lo trova salta la directory
                # Search for first txt-file
                # If not found skip
                try:
                    file = [p for p in run.iterdir() if p.suffix == ".txt"][0]
                except IndexError:
                    continue

                # A questo punto i dati della directory sono inseriti nel Database.
                # At this point directory datas are inserted in database.
                cur.execute(
                    f"""
                    INSERT INTO campioni (campione, date, misura)
                    VALUES ('{CAMPIONE}', '{TIME.date()}', '{RUN}');
                """
                )

                con.commit()
                # getting last row-id to update other relational tables
                row_id = cur.lastrowid  # Mi serve per aggiornare anche le altre tabelle

                read_file = open(file)

                line = read_file.readline()

                if len(line) < 4:  # controlla se ho un header
                    # Check if header is present
                    bImpulso = True
                    n_row_impulse = int(line)

                    pulse = header_in_table(
                        con=con,
                        cur=cur,
                        row_id=row_id,
                        n_row_impulse=n_row_impulse,
                        read_file=read_file,
                    )

                    GEN_REP = int(read_file.readline())

                else:
                    bImpulso = False
                    read_file.seek(0)  # Non ho un header mi riporto all'inizio del file
                    GEN_REP = 0  # Resetting this value
                    pulse = []  # If header is not present, go to file start.

                check = measure_in_table(
                    con=con, cur=cur, row_id=row_id, read_file=read_file
                )
                if check == 0: # Prevenzione contro misure fallite
                    cur.execute(f"DELETE FROM CAMPIONI WHERE ROWID = {row_id}")
                    cur.execute(f"DELETE FROM MISURE WHERE ROWID = {row_id}")
                    print(f"File Corrente Corrotto : {str(run)}")

                else:
                    TIPOLOGIA = check_type(b_pulse=bImpulso, pulse=pulse, check=check)

                    cur.execute(
                        f"UPDATE campioni SET tipologia = '{TIPOLOGIA}' WHERE rowid = {row_id}"
                    )

                    cur.execute(
                        f"UPDATE campioni SET gen_rep = '{GEN_REP}' WHERE rowid = {row_id}"
                    )
                con.commit()

                count+=1
                read_file.close()
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Program to automatically upgrade my database"
    )

    parser.add_argument("db_name", type=str, help="Database Name")

    parser.add_argument(
        "search_dir", type=str, nargs="+", help="directories to search for data."
    )

    parser.add_argument(
        "--reset",
        "-r",
        help="Reset Database, this flag empties the database and then recreate it",
        action="store_true",  # Remember this means default value is false
    )

    parser.add_argument(
        "--verbose",
        "-v",
        help = "enable verbose mode",
        action="store_true"
    )


    args = parser.parse_args()

    print("Checking Database")

    db_name = Path(args.db_name)

    if not db_name.exists():
        # print("db_name")
        con = create_db(db_name)
    else:
        print(db_name, " Exists")
        con = sqlite3.connect(db_name)

    # if Windows
    if os.name == "nt":
        search_dirs = []
        for j in args.search_dir:
            search_dirs.extend([i for i in Path(".").glob(j) if i.is_dir()])
    else:
        # These are directories to search for datas
        search_dirs = [Path(dirs) for dirs in args.search_dir if Path(dirs).is_dir()]

    print("Dirs found:")
    print("\n".join(str(item) for item in search_dirs))

    print("Creating Cursor")

    cur = con.cursor()

    # Attiva le seguenti linee per resettare il Database

    if args.reset:
        reset_db(con=con, cur=cur)
    ######

    print("Starting Uploading")

    n_uploaded = clean_and_upload(con=con, cur=cur, search_dirs=search_dirs)

    print(f"Uploaded {n_uploaded} files")

    # Keeping for debugging purpose.
    if args.verbose:
        res = cur.execute("SELECT * FROM campioni INNER JOIN impulsi ON campioni.rowid = impulsi.rowid")
        print(res.fetchall())

    con.commit()
    con.close()

    exit(0)


if __name__ == "__main__":
    main()
