import streamlit as st
import hexaly.optimizer
import math
import os
import pandas as pd
import matplotlib.pyplot as plt
import vrplib
import plotly.graph_objs as go
import matplotlib.cm as cm
import threading
import time
import sys
import contextlib
import re
import numpy as np
from io import StringIO
import random
import tempfile

# ----- Configuration de la page -----
st.set_page_config(page_title="VRPTW Solver", layout="wide")

# ----- Menu de navigation -----
page = st.sidebar.selectbox("Navigation", ["VRPTW Solver V1", "VRPTW Solver V2", "Tournes Similarity"])



if page == "VRPTW Solver V1":
    st.title("Résolution du Problème de VRP")
    st.markdown("Optimisez vos tournées et consultez la sortie du terminal, les itinéraires détaillés et la visualisation graphique via les onglets.")

    # ----- Fonctions de lecture -----
    def read_elem(filename):
        with open(filename) as f:
            return [str(elem) for elem in f.read().split()]

    def compute_dist(xi, xj, yi, yj):
        return math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)

    def compute_distance_matrix(customers_x, customers_y):
        nb_customers = len(customers_x)
        return [[0 if i == j else compute_dist(customers_x[i], customers_x[j], customers_y[i], customers_y[j])
                 for j in range(nb_customers)] for i in range(nb_customers)]

    def compute_distance_depots(depot_x, depot_y, customers_x, customers_y):
        return [compute_dist(depot_x, cx, depot_y, cy) for cx, cy in zip(customers_x, customers_y)]

    def read_input_cvrptw(filename):
        file_it = iter(read_elem(filename))
        for _ in range(4):
            next(file_it)
        nb_trucks = int(next(file_it))
        truck_capacity = int(next(file_it))
        for _ in range(13):
            next(file_it)
        depot_x = int(next(file_it))
        depot_y = int(next(file_it))
        for _ in range(2):
            next(file_it)
        max_horizon = int(next(file_it))
        next(file_it)

        customers_x, customers_y, demands = [], [], []
        earliest_start, latest_end, service_time = [], [], []
        while True:
            val = next(file_it, None)
            if val is None:
                break
            i = int(val) - 1
            customers_x.append(int(next(file_it)))
            customers_y.append(int(next(file_it)))
            demands.append(int(next(file_it)))
            ready = int(next(file_it))
            due = int(next(file_it))
            stime = int(next(file_it))
            earliest_start.append(ready)
            latest_end.append(due + stime)
            service_time.append(stime)
        nb_customers = i + 1

        distance_matrix = compute_distance_matrix(customers_x, customers_y)
        distance_depots = compute_distance_depots(depot_x, depot_y, customers_x, customers_y)

        return (nb_customers, nb_trucks, truck_capacity, distance_matrix, distance_depots,
                demands, service_time, earliest_start, latest_end, max_horizon)

    # ----- Fonction de résolution du vrptw -----
    def solve_vrp(instance_file, str_time_limit, output_file):
        nb_customers, nb_trucks, truck_capacity, dist_matrix_data, dist_depot_data, \
        demands_data, service_time_data, earliest_start_data, latest_end_data, max_horizon = read_input_cvrptw(instance_file)

        with hexaly.optimizer.HexalyOptimizer() as optimizer:
            model = optimizer.model
            customers_sequences = [model.list(nb_customers) for _ in range(nb_trucks)]
            model.constraint(model.partition(customers_sequences))

            demands = model.array(demands_data)
            earliest = model.array(earliest_start_data)
            latest = model.array(latest_end_data)
            service_time = model.array(service_time_data)
            dist_matrix = model.array(dist_matrix_data)
            dist_depot = model.array(dist_depot_data)

            dist_routes = [None] * nb_trucks
            end_time = [None] * nb_trucks
            home_lateness = [None] * nb_trucks
            lateness = [None] * nb_trucks

            trucks_used = [(model.count(customers_sequences[k]) > 0) for k in range(nb_trucks)]
            nb_trucks_used = model.sum(trucks_used)

            for k in range(nb_trucks):
                sequence = customers_sequences[k]
                c = model.count(sequence)
                route_quantity = model.sum(sequence, model.lambda_function(lambda j: demands[j]))
                model.constraint(route_quantity <= truck_capacity)

                dist_lambda = model.lambda_function(lambda i: model.at(dist_matrix, sequence[i - 1], sequence[i]))
                dist_routes[k] = model.sum(model.range(1, c), dist_lambda) + model.iif(c > 0,
                                    dist_depot[sequence[0]] + dist_depot[sequence[c - 1]], 0)

                end_time_lambda = model.lambda_function(
                    lambda i, prev: model.max(earliest[sequence[i]],
                                              model.iif(i == 0, dist_depot[sequence[0]], prev + model.at(dist_matrix, sequence[i - 1], sequence[i])))
                                  + service_time[sequence[i]])
                end_time[k] = model.array(model.range(0, c), end_time_lambda, 0)
                home_lateness[k] = model.iif(trucks_used[k],
                                              model.max(0, end_time[k][c - 1] + dist_depot[sequence[c - 1]] - max_horizon),
                                              0)
                late_lambda = model.lambda_function(lambda i: model.max(0, end_time[k][i] - latest[sequence[i]]))
                lateness[k] = home_lateness[k] + model.sum(model.range(0, c), late_lambda)

            total_lateness = model.sum(lateness)
            total_distance = model.div(model.round(100 * model.sum(dist_routes)), 100)

            model.minimize(total_lateness)
            model.minimize(nb_trucks_used)
            model.minimize(total_distance)
            model.close()

            optimizer.param.time_limit = int(str_time_limit)
            optimizer.solve()

            if output_file is not None:
                with open(output_file, 'w') as f:
                    f.write(f"{nb_trucks_used.value} {total_distance.value}\n")
                    for k in range(nb_trucks):
                        if trucks_used[k].value != 1:
                            continue
                        for customer in customers_sequences[k].value:
                            f.write(f"{customer + 1} ")
                        f.write("\n")
        return output_file

    # ----- Interface Utilisateur : Configuration dans la sidebar -----
    with st.sidebar:
        st.header("Configuration")
        uploaded_file = st.file_uploader("Télécharger l'instance Solomon", type=["txt"])
        str_time_limit = st.text_input("Temps limite d'exécution (secondes)", "20")
        output_file = st.text_input("Nom du fichier de sortie", "solution.txt")
        lancer = st.button("Lancer la résolution")

    log_buffer = StringIO()
    solver_output = None

    def run_solver():
        global solver_output, log_buffer
        with contextlib.redirect_stdout(log_buffer):
            solver_output = solve_vrp(instance_file, str_time_limit, output_file)

    if lancer:
        if uploaded_file is not None:
            st.info("Fichier téléchargé, résolution en cours...")
            # Sauvegarde du fichier téléchargé sur le disque
            instance_file = uploaded_file.name
            with open(instance_file, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Exécuter le solveur dans un thread séparé
            solver_thread = threading.Thread(target=run_solver)
            solver_thread.start()

            # Utiliser un spinner sans afficher le log en temps réel
            with st.spinner("Résolution en cours..."):
                solver_thread.join()

            st.success(f"Solution sauvegardée dans {solver_output}")

            # Création des onglets une fois la résolution terminée
            tabs = st.tabs(["Terminal", "Itinéraires détaillés", "Visualisation graphique"])

            # Onglet Terminal : affichage du log final à partir du mot "Model:"
            with tabs[0]:
                st.subheader("Résultats : ")
                final_log = log_buffer.getvalue()
                start_idx = final_log.find("Model:")
                if start_idx != -1:
                    final_log = final_log[start_idx:]
                st.text(final_log)

            # Lecture et traitement des résultats pour les autres onglets
            with open(solver_output, "r") as f:
                solution_data = f.read()
            lines = solution_data.splitlines()
            nb_trucks_used = int(lines[0].split()[0])
            total_distance = float(lines[0].split()[1])
            routes = [[int(x) for x in line.split()] for line in lines[1:]]

            # Onglet Itinéraires détaillés
            with tabs[1]:
                st.subheader("Itinéraires détaillés")
                itineraire_details = []
                for i, route in enumerate(routes):
                    trajet = [f"Dépôt -> Client {route[0] + 1}"]
                    for j in range(1, len(route)):
                        trajet.append(f"Client {route[j-1] + 1} -> Client {route[j] + 1}")
                    trajet.append(f"Client {route[-1] + 1} -> Dépôt")
                    itineraire_details.append({
                        "Camion": f"Camion {i + 1}",
                        "Itinéraire": " -> ".join(trajet),
                        "Nombre de Clients": len(route)
                    })
                df_itineraire = pd.DataFrame(itineraire_details)
                st.dataframe(df_itineraire, use_container_width=True)

                # Graphique du nombre de clients par camion
                st.subheader("Nombre de clients par camion")
                clients_per_truck = [len(route) for route in routes]
                fig_bar = go.Figure(data=[go.Bar(x=[f"Camion {i+1}" for i in range(len(clients_per_truck))],
                                                 y=clients_per_truck)])
                fig_bar.update_layout(title="Nombre de clients par camion",
                                      xaxis_title="Camion",
                                      yaxis_title="Nombre de Clients")
                st.plotly_chart(fig_bar, use_container_width=True)

            # Onglet Visualisation graphique
            with tabs[2]:
                st.subheader("Visualisations")
                # Lecture de l'instance
                instance = vrplib.read_instance(instance_file, instance_format="solomon")
                coords = instance["node_coord"]
                depot = coords[0]
                clients = coords[1:]

                # Graphique non orienté
                fig_no_arrows = go.Figure()

                # Le dépôt
                fig_no_arrows.add_trace(go.Scatter(
                    x=[depot[0]], y=[depot[1]],
                    mode='markers+text',
                    text=['Dépôt'],
                    textposition='top center',
                    marker=dict(color='red', size=15, symbol='x'),
                    name='Dépôt'
                ))

                # Les clients
                for i, client in enumerate(clients):
                    fig_no_arrows.add_trace(go.Scatter(
                        x=[client[0]], y=[client[1]],
                        mode='markers+text',
                        text=[f"{i+1}"],
                        textposition='top center',
                        marker=dict(color='blue', size=10),
                        name=f"Client {i+1}" if i == 0 else "",
                        showlegend=False
                    ))

                num_routes = len(routes)
                colors = cm.get_cmap('tab20', num_routes)

                # Tracer les itinéraires
                for i, route in enumerate(routes):
                    color = colors(i)
                    rgb_color = f'rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})'

                    # Liaison entre le dépôt et le premier client
                    fig_no_arrows.add_trace(go.Scatter(
                        x=[depot[0], coords[route[0]][0]],
                        y=[depot[1], coords[route[0]][1]],
                        mode='lines',
                        line=dict(color=rgb_color, width=2),
                        name=f"Itinéraire {i+1}"
                    ))

                    # Liaison entre les clients
                    for j in range(len(route)-1):
                        fig_no_arrows.add_trace(go.Scatter(
                            x=[coords[route[j]][0], coords[route[j+1]][0]],
                            y=[coords[route[j]][1], coords[route[j+1]][1]],
                            mode='lines',
                            line=dict(color=rgb_color, width=2),
                            showlegend=False
                        ))

                    # Liaison du dernier client au dépôt
                    fig_no_arrows.add_trace(go.Scatter(
                        x=[coords[route[-1]][0], depot[0]],
                        y=[coords[route[-1]][1], depot[1]],
                        mode='lines',
                        line=dict(color=rgb_color, width=2),
                        showlegend=False
                    ))

                fig_no_arrows.update_layout(
                    title="Itinéraires des camions",
                    xaxis_title="Coordonnée X",
                    yaxis_title="Coordonnée Y",
                    legend=dict(x=1.05, y=1, traceorder="normal", font=dict(family="sans-serif", size=12))
                )

                st.plotly_chart(fig_no_arrows, use_container_width=True)

                for trace in fig_no_arrows.data:
                    trace.showlegend = False

                # Graphique orienté
                fig_with_arrows = go.Figure(fig_no_arrows)

                # Ajout des flèches directionnelles
                for i, route in enumerate(routes):
                    color = colors(i)
                    rgb_color = f'rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})'

                    for j in range(len(route)-1):
                        fig_with_arrows.add_annotation(
                            x=coords[route[j+1]][0],
                            y=coords[route[j+1]][1],
                            ax=coords[route[j]][0],
                            ay=coords[route[j]][1],
                            xref="x",
                            yref="y",
                            axref="x",
                            ayref="y",
                            showarrow=True,
                            arrowhead=3,
                            arrowsize=1.5,
                            arrowwidth=1.5,
                            arrowcolor=rgb_color
                        )

                    # Ajout de la flèche directionnelle pour le retour au dépôt
                    fig_with_arrows.add_annotation(
                        x=depot[0],
                        y=depot[1],
                        ax=coords[route[-1]][0],
                        ay=coords[route[-1]][1],
                        xref="x",
                        yref="y",
                        axref="x",
                        ayref="y",
                        showarrow=True,
                        arrowhead=3,
                        arrowsize=1.5,
                        arrowwidth=1.5,
                        arrowcolor=rgb_color
                    )

                fig_with_arrows.update_layout(
                    title="Sens des tournées",
                    xaxis_title="Coordonnée X",
                    yaxis_title="Coordonnée Y",
                    legend=dict(x=1.05, y=1, traceorder="normal", font=dict(family="sans-serif", size=12))
                )

                st.plotly_chart(fig_with_arrows, use_container_width=True)
        else:
            st.error("Veuillez télécharger un fichier d'instance.")

elif page == "VRPTW Solver V2":
    st.title("Simulation du baisse de l'effectif des clients")

    import math
    import io
    import plotly.graph_objects as go
    import numpy as np
    import re
    import random

    # Fonction pour lire le fichier d'entrée
    def read_elem(file):
        return [str(elem) for elem in file.decode('utf-8').split()]

    # Fonction de simulation de suppression aléatoire des clients
    def simulate_customer_deletion(nb_customers, customers_x, customers_y, demands, earliest_start, latest_end, service_time, deletion_rate):
        new_customers_x = [customers_x[0]]
        new_customers_y = [customers_y[0]]
        new_demands = [demands[0]]
        new_earliest = [earliest_start[0]]
        new_latest = [latest_end[0]]
        new_service_time = [service_time[0]]
        kept_ids = [0]

        for i in range(1, nb_customers):
            if random.random() > deletion_rate:
                new_customers_x.append(customers_x[i])
                new_customers_y.append(customers_y[i])
                new_demands.append(demands[i])
                new_earliest.append(earliest_start[i])
                new_latest.append(latest_end[i])
                new_service_time.append(service_time[i])
                kept_ids.append(i)

        new_nb_customers = len(new_customers_x)
        return new_nb_customers, new_customers_x, new_customers_y, new_demands, new_earliest, new_latest, new_service_time, kept_ids

    # Fonction pour lire l'entrée avec option de suppression de clients
    def read_input_cvrptw(file, deletion_rate=0.0):
        file_it = iter(read_elem(file))

        for i in range(4):
            next(file_it)

        nb_trucks = int(next(file_it))
        truck_capacity = int(next(file_it))

        for i in range(13):
            next(file_it)

        depot_x = int(next(file_it))
        depot_y = int(next(file_it))

        for i in range(2):
            next(file_it)

        max_horizon = int(next(file_it))
        next(file_it)

        customers_x, customers_y, demands, earliest_start, latest_end, service_time = [], [], [], [], [], []

        while True:
            val = next(file_it, None)
            if val is None:
                break
            i = int(val) - 1
            customers_x.append(int(next(file_it)))
            customers_y.append(int(next(file_it)))
            demands.append(int(next(file_it)))
            ready = int(next(file_it))
            due = int(next(file_it))
            stime = int(next(file_it))
            earliest_start.append(ready)
            latest_end.append(due + stime)
            service_time.append(stime)

        nb_customers = i + 1

        if deletion_rate > 0.0:
            nb_customers, customers_x, customers_y, demands, earliest_start, latest_end, service_time, kept_ids = simulate_customer_deletion(
                nb_customers, customers_x, customers_y, demands, earliest_start, latest_end, service_time, deletion_rate
            )
        else:
            kept_ids = list(range(nb_customers))

        distance_matrix = compute_distance_matrix(customers_x, customers_y)
        distance_depots = compute_distance_depots(depot_x, depot_y, customers_x, customers_y)

        return nb_customers, nb_trucks, truck_capacity, distance_matrix, distance_depots, \
            demands, service_time, earliest_start, latest_end, max_horizon, kept_ids

    def compute_distance_matrix(customers_x, customers_y):
        nb_customers = len(customers_x)
        distance_matrix = [[None for _ in range(nb_customers)] for _ in range(nb_customers)]
        for i in range(nb_customers):
            distance_matrix[i][i] = 0
            for j in range(nb_customers):
                dist = compute_dist(customers_x[i], customers_x[j], customers_y[i], customers_y[j])
                distance_matrix[i][j] = dist
                distance_matrix[j][i] = dist
        return distance_matrix

    def compute_distance_depots(depot_x, depot_y, customers_x, customers_y):
        nb_customers = len(customers_x)
        distance_depots = [None] * nb_customers
        for i in range(nb_customers):
            dist = compute_dist(depot_x, customers_x[i], depot_y, customers_y[i])
            distance_depots[i] = dist
        return distance_depots

    def compute_dist(xi, xj, yi, yj):
        return math.sqrt((xi - xj)**2 + (yi - yj)**2)

    # Fonction pour résoudre le VRP
    def main(instance_file, str_time_limit, output_file, deletion_rate=0.0):
        nb_customers, nb_trucks, truck_capacity, dist_matrix_data, dist_depot_data, \
            demands_data, service_time_data, earliest_start_data, latest_end_data, \
            max_horizon, kept_ids = read_input_cvrptw(instance_file, deletion_rate)

        with hexaly.optimizer.HexalyOptimizer() as optimizer:
            model = optimizer.model

            customers_sequences = [model.list(nb_customers) for _ in range(nb_trucks)]
            model.constraint(model.partition(customers_sequences))

            demands = model.array(demands_data)
            earliest = model.array(earliest_start_data)
            latest = model.array(latest_end_data)
            service_time = model.array(service_time_data)
            dist_matrix = model.array(dist_matrix_data)
            dist_depot = model.array(dist_depot_data)

            dist_routes = [None] * nb_trucks
            end_time = [None] * nb_trucks
            home_lateness = [None] * nb_trucks
            lateness = [None] * nb_trucks

            trucks_used = [(model.count(customers_sequences[k]) > 0) for k in range(nb_trucks)]
            nb_trucks_used = model.sum(trucks_used)

            for k in range(nb_trucks):
                sequence = customers_sequences[k]
                c = model.count(sequence)

                demand_lambda = model.lambda_function(lambda j: demands[j])
                route_quantity = model.sum(sequence, demand_lambda)
                model.constraint(route_quantity <= truck_capacity)

                dist_lambda = model.lambda_function(lambda i: model.at(dist_matrix, sequence[i - 1], sequence[i]))
                dist_routes[k] = model.sum(model.range(1, c), dist_lambda) \
                    + model.iif(c > 0, dist_depot[sequence[0]] + dist_depot[sequence[c - 1]], 0)

                end_time_lambda = model.lambda_function(
                    lambda i, prev: model.max(earliest[sequence[i]],
                                                model.iif(i == 0, dist_depot[sequence[0]],
                                                            prev + model.at(dist_matrix, sequence[i - 1], sequence[i]))) + service_time[sequence[i]])
                end_time[k] = model.array(model.range(0, c), end_time_lambda, 0)

                home_lateness[k] = model.iif(
                    trucks_used[k],
                    model.max(0, end_time[k][c - 1] + dist_depot[sequence[c - 1]] - max_horizon),
                    0)

                late_lambda = model.lambda_function(lambda i: model.max(0, end_time[k][i] - latest[sequence[i]]))
                lateness[k] = home_lateness[k] + model.sum(model.range(0, c), late_lambda)

            total_lateness = model.sum(lateness)
            total_distance = model.div(model.round(100 * model.sum(dist_routes)), 100)

            model.minimize(total_lateness)
            model.minimize(nb_trucks_used)
            model.minimize(total_distance)

            model.close()

            optimizer.param.time_limit = int(str_time_limit)
            optimizer.solve()

            if output_file:
                output_solution(output_file, nb_trucks_used, total_distance, customers_sequences, end_time, service_time_data, dist_depot_data, kept_ids)

    # Fonction d'écriture la solution dans un fichier .txt
    def output_solution(output_file, nb_trucks_used, total_distance, customers_sequences, end_time, service_time_data, dist_depot_data, kept_ids):
        with open(output_file, 'w') as f:
            f.write("%d %d\n" % (nb_trucks_used.value, total_distance.value))
            for k in range(len(customers_sequences)):
                seq = customers_sequences[k].value
                if seq is None or len(seq) == 0:
                    continue
                for i, customer in enumerate(seq):
                    arrival_time = end_time[k].value[i] - service_time_data[customer]
                    f.write("%d(%d) " % (kept_ids[customer] + 1, arrival_time))
                last_index = len(seq) - 1
                last_customer = seq[last_index]
                return_time = end_time[k].value[last_index] + dist_depot_data[last_customer]
                f.write("0(%d)" % return_time)
                f.write("\n")

    # Fonction pour charger les données brutes depuis un fichier uploadé
    def load_raw_data(file):
        lines = file.getvalue().decode("utf-8").splitlines()
        start_index = lines.index("CUSTOMER") + 2
        data = {}
        depot_coords = None
        for line in lines[start_index:]:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            cust_no, x, y, demand, ready_time, due_date, service_time = map(int, parts[:7])
            if cust_no == 0:
                depot_coords = (x, y)
            data[cust_no] = (x, y, ready_time, due_date)
        return data, depot_coords

    # Fonction pour charger la solution depuis un fichier uploadé
    def load_solution(file):
        lines = file.getvalue().decode("utf-8").splitlines()
        solution = []
        for line in lines[1:]:
            route = re.findall(r'\d+\(\d+\)', line)
            parsed_route = [(int(r.split('(')[0]), int(r.split('(')[1][:-1])) for r in route]
            solution.append(parsed_route)
        return solution

    def plot_interactive_gantt_chart(data, solution, deletion_rate):
        fig = go.Figure()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']

        for i, route in enumerate(solution):
            for cust_no, passage_time in route:
                x, y, ready_time, due_date = data[cust_no]
                duration = due_date - ready_time
                if cust_no != 0:
                    fig.add_trace(go.Bar(
                        x=[duration],
                        y=[f'Vehicule {i+1}'],
                        orientation='h',
                        base=ready_time,
                        name=f'Vehicule {i+1}',
                        marker=dict(color=colors[i % len(colors)], opacity=0.6),
                        hoverinfo='x+y'
                    ))
                if cust_no != 0:
                    fig.add_trace(go.Scatter(
                        x=[passage_time],
                        y=[f'Vehicule {i+1}'],
                        mode='markers',
                        marker=dict(color='white', size=8, symbol='circle'),
                        showlegend=False,
                        hoverinfo='text',
                        text=[f"N° Client : {cust_no}<br>Fenêtre temporelle : {ready_time} - {due_date}<br>Heure de passage : {passage_time}"]
                    ))

        title = "Diagramme de Gantt des Tournées - Plein Service" if deletion_rate == 0 else f"Diagramme de Gantt des Tournées - {int(deletion_rate * 100)}% de baisse"
        fig.update_layout(
            title=title,
            xaxis_title="Temps",
            yaxis_title="Véhicule",
            showlegend=True,
            barmode='stack',
            height=600,
            template='plotly_dark'
        )
        return fig

    def plot_time_space_plotly(data, depot_coords, solution, deletion_rate):
        fig = go.Figure()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']

        depot_ready_time = data[0][2]
        depot_due_date = data[0][3]
        fig.add_shape(
            type='line',
            x0=depot_ready_time,
            y0=0,
            x1=depot_due_date,
            y1=0,
            line=dict(color='red', dash='dash', width=2),
            xref='x',
            yref='y'
        )

        for i, route in enumerate(solution):
            time_vals = []
            distance_vals = []
            tooltip_text = []
            route_with_depot = [(0, 0)] + route
            for cust_no, passage_time in route_with_depot:
                if cust_no == 0:
                    x = depot_coords[0]
                    y = depot_coords[1]
                    tooltip_text.append("Dépôt")
                else:
                    x, y, ready_time, due_date = data[cust_no]
                    tooltip_text.append(
                        f"N° Client : {cust_no}<br>"
                        f"Fenêtre temporelle : {ready_time} - {due_date}<br>"
                        f"Heure de passage : {passage_time}"
                    )
                    distance = np.sqrt((x - depot_coords[0])**2 + (y - depot_coords[1])**2)
                    fig.add_shape(
                        type='line',
                        x0=ready_time,
                        y0=distance,
                        x1=due_date,
                        y1=distance,
                        line=dict(color='gray', dash='dash', width=2),
                        xref='x',
                        yref='y'
                    )
                distance = np.sqrt((x - depot_coords[0])**2 + (y - depot_coords[1])**2)
                time_vals.append(passage_time)
                distance_vals.append(distance)

            fig.add_trace(go.Scatter(
                x=time_vals,
                y=distance_vals,
                mode='lines+markers',
                name=f'Tournée {i+1}',
                line=dict(color=colors[i % len(colors)]),
                marker=dict(color=colors[i % len(colors)], size=8),
                hoverinfo='text',
                text=tooltip_text
            ))

        title = "Graphique Espace-Temps des Tournées - Plein Service" if deletion_rate == 0 else f"Graphique Espace-Temps des Tournées - {int(deletion_rate * 100)}% de baisse"
        fig.update_layout(
            title=title,
            xaxis_title="Temps",
            yaxis_title="Distance euclidienne au dépôt",
            legend_title="Tournées",
            template='plotly_dark',
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True)
        )
        return fig

    def plot_route_map(data, depot_coords, solution, deletion_rate):
        fig = go.Figure()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']

        fig.add_trace(go.Scatter(
            x=[depot_coords[0]],
            y=[depot_coords[1]],
            mode='markers+text',
            name='Dépôt',
            marker=dict(color='red', size=10),
            textposition='top center',
            hoverinfo='text'
        ))

        for i, route in enumerate(solution):
            x_vals = []
            y_vals = []
            tooltip_text = []
            client_labels = []
            route_with_depot = [(0, 0)] + route
            for cust_no, passage_time in route_with_depot:
                if cust_no == 0:
                    x = depot_coords[0]
                    y = depot_coords[1]
                    tooltip_text.append("Dépôt")
                    client_labels.append("Dépôt")
                else:
                    x, y, ready_time, due_date = data[cust_no]
                    tooltip_text.append(
                        f"N° Client : {cust_no}<br>"
                        f"Fenêtre temporelle : {ready_time} - {due_date}<br>"
                        f"Heure de passage : {passage_time}"
                    )
                    client_labels.append(str(cust_no))
                x_vals.append(x)
                y_vals.append(y)

            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines+markers+text',
                name=f'Tournée {i+1}',
                line=dict(color=colors[i % len(colors)]),
                marker=dict(color=colors[i % len(colors)], size=8),
                hoverinfo='text',
                text=client_labels,
                textposition='top center',
                textfont=dict(size=10)
            ))

        title = "Carte des Tournées - Plein Service" if deletion_rate == 0 else f"Carte des Tournées - {int(deletion_rate * 100)}% de baisse"
        fig.update_layout(
            title=title,
            xaxis_title="Coordonnée X",
            yaxis_title="Coordonnée Y",
            legend_title="Tournées",
            template='plotly_dark',
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True)
        )
        return fig

    # Ma nouvelle fonction pour afficher la position de chaque véhicule ainsi que les clients et les trajets suivis
    def plot_vehicle_positions(data, depot_coords, solution, deletion_rate):
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']

        # Calcul de la borne supérieure du temps global
        global_t_end = 0
        for route in solution:
            if route:
                t_last = route[-1][1]
                if t_last > global_t_end:
                    global_t_end = t_last
        time_step = 1  # pas de temps pour l'animation

        # Calcul des positions interpolées pour chaque véhicule le long de sa tournée
        vehicle_positions = {}  # véhicule -> liste de tuples (temps, x, y)
        for i, route in enumerate(solution):
            # Ajout du dépôt en début de tournée
            route_full = [(0, 0)] + route
            pos_list = []
            for j in range(len(route_full) - 1):
                cust1, t1 = route_full[j]
                cust2, t2 = route_full[j + 1]
                if cust1 == 0:
                    x1, y1 = depot_coords
                else:
                    x1, y1, _, _ = data[cust1]
                if cust2 == 0:
                    x2, y2 = depot_coords
                else:
                    x2, y2, _, _ = data[cust2]
                # Création d'échantillons entre t1 et t2
                if t2 - t1 == 0:
                    sample_times = [t1]
                else:
                    sample_times = list(np.arange(t1, t2, time_step))
                    if sample_times[-1] != t2:
                        sample_times.append(t2)
                for t in sample_times:
                    frac = 0 if (t2 - t1) == 0 else (t - t1) / (t2 - t1)
                    x = x1 + frac * (x2 - x1)
                    y = y1 + frac * (y2 - y1)
                    pos_list.append((t, x, y))
            if route_full:
                final_time = route_full[-1][1]
                if final_time < global_t_end:
                    pos_list.append((global_t_end, depot_coords[0], depot_coords[1]))
            vehicle_positions[i] = pos_list

        # Construction des traces statiques : clients et trajectoires complètes
        static_traces = []
        # Pour chaque tournée, tracer la trajectoire reliant les points de passage (avec dépôt inclus)
        for i, route in enumerate(solution):
            route_full = [(0, 0)] + route
            xs, ys, labels = [], [], []
            for cust, _ in route_full:
                if cust == 0:
                    x, y = depot_coords
                    labels.append("Dépôt")
                else:
                    x, y, _, _ = data[cust]
                    labels.append(str(cust))
                xs.append(x)
                ys.append(y)
            static_traces.append(go.Scatter(
                x=xs,
                y=ys,
                mode='lines+markers+text',
                marker=dict(size=8, color=colors[i % len(colors)]),
                line=dict(dash='dash', color=colors[i % len(colors)]),
                name=f'Tournée {i+1}',
                text=labels,
                textposition='top center'
            ))
        # Trace statique de tous les clients (en arrière-plan)
        customer_x, customer_y, cust_labels = [], [], []
        for cust_id, info in data.items():
            if cust_id != 0:
                customer_x.append(info[0])
                customer_y.append(info[1])
                cust_labels.append(str(cust_id))
        static_traces.append(go.Scatter(
            x=customer_x,
            y=customer_y,
            mode='markers',
            marker=dict(color='white', size=6, symbol='circle'),
            name='Clients',
            hoverinfo='text',
            text=cust_labels
        ))
        # Trace statique du dépôt (si non déjà inclus)
        static_traces.append(go.Scatter(
            x=[depot_coords[0]],
            y=[depot_coords[1]],
            mode='markers+text',
            marker=dict(size=12, color='red'),
            name='Dépôt',
            text=['Dépôt'],
            textposition='bottom center'
        ))

        # Construction des traces dynamiques pour la position des véhicules
        dynamic_traces_initial = []
        for i, pos_list in vehicle_positions.items():
            # Position initiale (t = 0)
            pos = depot_coords
            for (pt, x, y) in pos_list:
                if pt <= 0:
                    pos = (x, y)
                else:
                    break
            dynamic_traces_initial.append(go.Scatter(
                x=[pos[0]],
                y=[pos[1]],
                mode='markers+text',
                marker=dict(size=12, color=colors[i % len(colors)]),
                name=f'Vehicule {i+1}',
                text=[f'Vehicule {i+1}'],
                textposition='top center',
                showlegend=True,
                hoverinfo='text'
            ))
        # Combinaison des traces statiques et dynamiques pour l'état initial
        initial_data = static_traces + dynamic_traces_initial

        # Construction des frames pour l'animation : on conserve les traces statiques et on met à jour la position des véhicules
        frame_times = np.arange(0, global_t_end + time_step, time_step)
        frames = []
        for t in frame_times:
            dynamic_traces_frame = []
            for i, pos_list in vehicle_positions.items():
                pos = depot_coords
                for (pt, x, y) in pos_list:
                    if pt <= t:
                        pos = (x, y)
                    else:
                        break
                dynamic_traces_frame.append(go.Scatter(
                    x=[pos[0]],
                    y=[pos[1]],
                    mode='markers+text',
                    marker=dict(size=12, color=colors[i % len(colors)]),
                    name=f'Vehicule {i+1}',
                    text=[f'Vehicule {i+1}'],
                    textposition='top center',
                    hoverinfo='text'
                ))
            # La frame complète inclut les traces statiques (inchangées) + les traces dynamiques mises à jour
            frame_data = static_traces + dynamic_traces_frame
            frames.append(go.Frame(data=frame_data, name=str(t)))

        fig = go.Figure(
            data=initial_data,
            layout=go.Layout(
                title="Animation",
                xaxis=dict(title="Coordonnée X"),
                yaxis=dict(title="Coordonnée Y"),
                updatemenus=[{
                    "type": "buttons",
                    "buttons": [{
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]
                    }]
                }],
                sliders=[{
                    "steps": [{
                        "method": "animate",
                        "label": str(t),
                        "args": [[str(t)], {"frame": {"duration": 500, "redraw": True}, "mode": "immediate"}]
                    } for t in frame_times],
                    "currentvalue": {"prefix": "Temps: "}
                }]
            ),
            frames=frames
        )
        return fig

    # Application Streamlit
    def streamlit_app():
        uploaded_file = st.file_uploader("Upload the input instance file", type=["txt"])

        if uploaded_file is not None:
            instance_file = uploaded_file.read()
            st.text_area("Instance Data", str(instance_file.decode('utf-8')), height=300)

            str_time_limit = st.text_input("Temps limite d'exécution (secondes)", "20")
            output_file = st.text_input("Nom du fichier de sortie", "solution.txt")
            deletion_rate = st.slider("Taux de suppression des clients (0 = aucune suppression, 1 = suppression totale)", 0.0, 1.0, 0.0, step=0.05)

            if st.button("Lancer la résolution"):
                with st.spinner('Solving the VRP...'):
                    main(instance_file, str_time_limit, output_file, deletion_rate)
                st.success(f"Solution saved to {output_file}")

                with open(output_file, "r") as f:
                    solution_data = f.read()
                st.text_area("Solution", solution_data, height=300)

                data, depot_coords = load_raw_data(uploaded_file)
                solution = load_solution(io.BytesIO(solution_data.encode('utf-8')))

                st.subheader("Diagramme de Gantt")
                fig_gantt = plot_interactive_gantt_chart(data, solution, deletion_rate)
                st.plotly_chart(fig_gantt)

                st.subheader("Graphique Espace-Temps")
                fig_time_space = plot_time_space_plotly(data, depot_coords, solution, deletion_rate)
                st.plotly_chart(fig_time_space)

                st.subheader("Carte des Tournées")
                fig_route_map = plot_route_map(data, depot_coords, solution, deletion_rate)
                st.plotly_chart(fig_route_map)

                st.subheader("Position de chaque véhicule à chaque instant t")
                fig_positions = plot_vehicle_positions(data, depot_coords, solution, deletion_rate)
                st.plotly_chart(fig_positions)

    if __name__ == "__main__":
        streamlit_app()


elif page == "Tournes Similarity":
    import plotly.express as px
    import numpy as np
    import pandas as pd
    import streamlit as st
    import plotly.express as px

    def parse_solution(file):
        """Extrait les tournées des fichiers de solution en ignorant le client 0."""
        tours = []
        lines = file.read().decode("utf-8").strip().split("\n")[1:]  # Ignorer la première ligne
        for line in lines:
            clients = [int(part.split("(")[0]) for part in line.split() if "(" in part]
            clients = set(clients) - {0}  # Ignorer le client 0
            tours.append(clients)
        return tours

    def compute_jaccard_matrix(tours1, tours2):
        """Calcule la matrice d'indice de Jaccard entre les tournées."""
        n, m = len(tours1), len(tours2)
        matrix = np.zeros((n, m))

        for i in range(n):
            for j in range(m):
                intersection = len(tours1[i] & tours2[j])  # Clients en commun
                union = len(tours1[i] | tours2[j])  # Clients différents
                matrix[i, j] = intersection / union if union != 0 else 0

        return matrix

    def rearrange_matrix(matrix):
        """Réorganise la matrice pour maximiser les indices sur la diagonale descendante."""
        n, m = matrix.shape
        used_rows = set()
        used_cols = set()
        
        row_order = [-1] * min(n, m)
        col_order = [-1] * min(n, m)

        # Associer chaque ligne à la colonne avec le max de Jaccard
        pairs = []
        for i in range(n):
            j_max = np.argmax(matrix[i, :])  # Trouver la meilleure correspondance
            pairs.append((i, j_max, matrix[i, j_max]))

        # Trier par indice de Jaccard décroissant
        pairs.sort(key=lambda x: x[2], reverse=True)

        for i, j, _ in pairs:
            if i not in used_rows and j not in used_cols:
                row_order[len(used_rows)] = i
                col_order[len(used_cols)] = j
                used_rows.add(i)
                used_cols.add(j)

        # Ajouter les tournées restantes
        remaining_rows = [i for i in range(n) if i not in used_rows]
        remaining_cols = [j for j in range(m) if j not in used_cols]
        
        row_order.extend(remaining_rows)
        col_order.extend(remaining_cols)

        # Filtrer les -1
        row_order = [x for x in row_order if x != -1]
        col_order = [x for x in col_order if x != -1]

        # Réordonner la matrice
        reordered_matrix = matrix[np.ix_(row_order, col_order)]
        
        return reordered_matrix

    # -------------------- INTERFACE STREAMLIT --------------------
    st.title("Comparaison de tournées avec l'indice de Jaccard")

    file1 = st.file_uploader("Charger le fichier solution AVANT suppression", type=["txt"])
    file2 = st.file_uploader("Charger le fichier solution APRÈS suppression", type=["txt"])

    if file1 and file2:
        tours1 = parse_solution(file1)
        tours2 = parse_solution(file2)

        # Calcul de la matrice de Jaccard
        jaccard_matrix = compute_jaccard_matrix(tours1, tours2)

        # Réorganisation de la matrice
        reordered_matrix = rearrange_matrix(jaccard_matrix)

        # Conversion en DataFrame pour affichage
        n_rows, n_cols = reordered_matrix.shape
        df_reordered = pd.DataFrame(
            reordered_matrix, 
            index=[f"Tour {i+1}" for i in range(n_rows)],  # Numérotation 1 à n
            columns=[f"Tour {j+1}" for j in range(n_cols)]  # Numérotation 1 à n
        )

        st.subheader("Matrice d'indice de similarité de Jaccard (Réorganisée)")
        st.dataframe(df_reordered)

        # Affichage sous forme de heatmap avec Plotly
        fig = px.imshow(
            reordered_matrix,
            labels=dict(color="Jaccard Index"),
            x=[f"Tour {j+1}" for j in range(n_cols)],  # 1 à n
            y=[f"Tour {i+1}" for i in range(n_rows)],  # 1 à n
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Viridis"
        )

        # Ajout des noms des axes
        fig.update_layout(
            xaxis_title="Tournées après suppression",
            yaxis_title="Tournées avant suppression"
        )

        st.plotly_chart(fig)
