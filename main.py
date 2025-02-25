import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

bean_display=False

def fragmentation(n_iter, split_ratio_range, initial_volume, epsilon):
    particles = initial_volume
    fine_particles = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    batch_size = 1000  # Process particles in batches
    fine_vol = 0.000005
    fine_vol_std = 0.6

    for i in range(n_iter):
        progress = i + 1
        progress_bar.progress(progress, text=f'Processing iteration {progress} of {n_iter}')
        status_text.text(f'Step {progress} of {n_iter} iterations')

        new_particles = []
        for j in range(0, len(particles), batch_size):
            batch = particles[j:j + batch_size]
            
            # Vectorized operations for the batch
            loss = batch * epsilon
            vol_remaining = batch - loss
            
            # Split remaining volume
            r = np.random.uniform(split_ratio_range, 1-split_ratio_range, size=len(batch))
            vol1 = r * vol_remaining
            vol2 = (1 - r) * vol_remaining
            new_particles.extend(np.concatenate([vol1, vol2]))

            # Generate fine particles
            n_fine = np.maximum(1, np.random.poisson(loss / fine_vol))
            total_fines = int(np.sum(n_fine))
            if total_fines > 0:
                fine_volumes = 10**(np.random.normal(np.log10(fine_vol), fine_vol_std, total_fines))
                fine_particles.extend(fine_volumes)

        particles = np.array(new_particles, dtype=np.float32)  # Use float32 instead of float64
        
        # Periodically clear lists to free memory
        new_particles = []
        if len(fine_particles) > 100000:
            fine_particles = np.array(fine_particles, dtype=np.float32).tolist()

    progress_bar.empty()
    status_text.empty()

    # Convert to final arrays using float32
    particles_arr = np.array(particles, dtype=np.float32)
    fines_arr = np.array(fine_particles, dtype=np.float32)
    all_particles = np.concatenate([particles_arr, fines_arr])

    return all_particles

def calculate_distribution(all_particles, num_bin=50, weight_type='surface'):
    diameters = (6 * all_particles / np.pi) ** (1/3)
    log_size = np.log10(diameters)
    bins = np.linspace(log_size.min(), log_size.max(), num_bin)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    if weight_type == 'volume':
        weights = all_particles
    elif weight_type == 'quantity':
        weights = None
    else:
        weights = np.pi * diameters**2

    hist, _ = np.histogram(log_size, bins=bins, weights=weights, density=True)
    return bin_centers, hist

def coffee_beans(num_beans=150, mean_diameter=4.2, std_dev=0.5, display=False):
    diameters = np.random.normal(mean_diameter, std_dev, num_beans)
    radii = np.abs(diameters / 2)
    volumes = (4/3) * np.pi * (radii)**3

    if display == True:
        st.write(f"Average diameters: {np.mean(diameters):.2f} ± {np.std(diameters):.2f} mm") 
        st.write(f"Average volume: {np.mean(volumes):.4f} ± {np.std(volumes):.4f} mm³") 
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))

        axs[0].hist(diameters, bins=15, color='#6f4e37', edgecolor='black')
        axs[0].set_title('Coffee Bean Size (diameter) Distribution')
        axs[0].set_xlabel('Size (mm)')
        axs[0].set_ylabel('Count')

        axs[1].hist(volumes, bins=15, color='#6f4e37', edgecolor='black')
        axs[1].set_title('Coffee Bean Volume Distribution')
        axs[1].set_xlabel('Volume (mm³)')

        plt.tight_layout()
        st.pyplot(fig)

    return diameters, volumes

st.title('HMS lab')

with st.sidebar:
    st.header("Simulation Parameters")
    
    uploaded_file = st.file_uploader("Upload experimental data (CSV)", type=['csv'])
    
    weight_type = st.radio(
        "Weight Type",
        options=['surface', 'volume', 'quantity'],
        index=0,
        help="Choose the weight type for distribution calculation"
    )
    
    n_iter = st.number_input('Number of Iterations', 
                            min_value=5, 
                            max_value=20, 
                            value=14,
                            step=1)
    
    split_ratio_range = st.number_input('Split Ratio Range',
                                       min_value=0.1,
                                       max_value=0.3,
                                       value=0.18,
                                       format="%.3f")
    
    epsilon = st.number_input('Epsilon (loss factor)',
                             min_value=0.001,
                             max_value=0.01,
                             value=0.0043,
                             format="%.4f")
    
    show_beans = st.checkbox('Show Coffee Bean Distribution', value=False)

if show_beans:
    st.header('Coffee Bean Initial Distribution')
    bean_display=True

if uploaded_file is not None:
    try:
        exp_data_raw = np.loadtxt(uploaded_file, delimiter=',', encoding='utf-8-sig')
        exp_data = exp_data_raw[(exp_data_raw[:, 0] >= 10) & (exp_data_raw[:, 0] <= 1000)]
        exp_data_x = exp_data[:, 0]/1000
        exp_data_y = exp_data[:, 1]
        exp_data_main = exp_data_raw[(exp_data_raw[:, 0] >= 100) & (exp_data_raw[:, 0] <= 1000)]
        
        st.markdown("""
    <style>
    .centered-header {
        text-align: center;
        font-size: 1.5em;
        padding: 15px;
    }
    </style>
    <h2 class='centered-header'>Data Plot</h2>
""", unsafe_allow_html=True)
        fig_main, ax_main = plt.subplots()
        plt.semilogx(exp_data_x, exp_data_y/np.max(exp_data_y), marker='o', color='red')
        ax_main.set_xlabel('Size, Diameter [mm]')
        ax_main.set_ylabel('Normalized value')
        ax_main.grid(True, which='both', linestyle='--', linewidth=0.5)
        st.pyplot(fig_main)
        
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        st.error("Please ensure your CSV file has the correct format (two columns: size and measurement).")
else:
    st.info("📤 Please upload your experimental data file (CSV) in the sidebar to compare with simulation results.")

st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #98FF98;
        color: black;
        border: 4px solid #228B22;
    }
    div.stButton > button:hover {
        background-color: #7FFF7F;
        color: black;
        border: 4px solid #006400;
    }
    </style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    start_button = st.button(
        'Start Simulation', 
        use_container_width=True,
        type="primary"
    )

if start_button:
    st.markdown("""
        <style>
        .centered-header {
            text-align: center;
            font-size: 2em;
            padding: 20px;
        }
        </style>
        <h1 class='centered-header'>Experimental and Simulation Results</h1>
    """, unsafe_allow_html=True)
    
    _, beans = coffee_beans(150, 4.2, 0.5)
    particles = fragmentation(n_iter, split_ratio_range, beans, epsilon)
    bin_centers, density = calculate_distribution(particles, num_bin=50, weight_type=weight_type)
    sim_data_raw = np.column_stack((10**bin_centers, density))
    sim_data = sim_data_raw[(sim_data_raw[:, 0] >= 0.008) & (sim_data_raw[:, 0] <= 1)]
    sim_data_main = sim_data_raw[(sim_data_raw[:, 0] >= 0.1) & (sim_data_raw[:, 0] <= 1)]

    fig_main, ax_main = plt.subplots()
    plt.semilogx(exp_data_x, exp_data_y/np.max(exp_data_main[:,1]), marker='o', color='red')
    plt.semilogx(sim_data[:,0], sim_data[:,1]/np.max(sim_data_main[:,1]), marker='.', color='skyblue')
    ax_main.set_xlabel('Size, Diameter [mm]')
    ax_main.set_ylabel('Normalized value')
    ax_main.legend(['Measured', 'Simulated'])
    ax_main.grid(True, which='both', linestyle='--', linewidth=0.5)
    st.pyplot(fig_main)
