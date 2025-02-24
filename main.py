import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

bean_display=False
# Define the Gaussian function
def double_gaussian(x, a1, b1, c1, a2, b2, c2):
    return (a1 * np.exp(-((x - b1) ** 2) / (2 * c1 ** 2)) +
            a2 * np.exp(-((x - b2) ** 2) / (2 * c2 ** 2)))

def double_gaussian_par(popt):
    a1, b1, c1, a2, b2, c2 = popt
    st.write("[ Fitted Parameters ]") # Use st.write instead of print
    st.write("Gaussian 1:")
    st.write(f"  amplitude = {a1:.4f}")
    st.write(f"  mean (mm) = {10**b1:.4f}")
    st.write(f"  std dev   = {c1:.4f}")
    st.write("Gaussian 2:")
    st.write(f"  amplitude = {a2:.4f}")
    st.write(f"  mean (mm) = {10**b2:.4f}")
    st.write(f"  std dev   = {c2:.4f}")

# Define the function to simulate the fragmentation process
def fragmentation(n_iter, split_ratio_range, initial_volume, epsilon):
    particles = initial_volume
    fine_particles = []

    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Fragmentation process: each particle is split into two fragments.
    for i in range(n_iter):
        # Update progress bar
        progress = (i + 1) / n_iter
        progress_bar.progress(progress)
        status_text.text(f'Processing iteration {i + 1} of {n_iter}')

        new_particles = []
        for vol in particles:
            loss = vol * epsilon
            vol_remaining = vol - loss

            # Split remaining volume
            r = np.random.uniform(split_ratio_range, 1-split_ratio_range)
            vol1 = r * vol_remaining
            vol2 = (1 - r) * vol_remaining
            new_particles.extend([vol1, vol2])

            fine_vol = 0.000005
            fine_vol_std = 0.6
            n_fine = max(1, np.random.poisson(loss / fine_vol))
            fine_volumes = 10**(np.random.normal(np.log10(fine_vol), fine_vol_std, n_fine))
            fine_particles.extend(fine_volumes.tolist())
        particles = new_particles

    # Clear the progress bar and status text when done
    progress_bar.empty()
    status_text.empty()

    # The particles in the bowl
    particles_arr = np.array(particles)
    fines_arr = np.array(fine_particles)
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
    else:  # surface area
        weights = np.pi * diameters**2

    hist, _ = np.histogram(log_size, bins=bins, weights=weights)
    density = hist / hist.sum() if weights is not None else hist / len(all_particles)
    return bin_centers, density

def coffee_beans(num_beans=150, mean_diameter=4.2, std_dev=0.5, display=False):

    diameters = np.random.normal(mean_diameter, std_dev, num_beans)
    radii = np.abs(diameters / 2)
    volumes = (4/3) * np.pi * (radii)**3

    # Print summary statistics
    # st.write(f"Average diameters: {np.mean(diameters):.2f} ± {np.std(diameters):.2f} mm") # Use st.write instead of print
    # st.write(f"Average volume: {np.mean(volumes):.4f} ± {np.std(volumes):.4f} mm³") # Use st.write instead of print
    if display == True:
        # Plot distributions
        st.write(f"Average diameters: {np.mean(diameters):.2f} ± {np.std(diameters):.2f} mm") 
        st.write(f"Average volume: {np.mean(volumes):.4f} ± {np.std(volumes):.4f} mm³") 
        fig, axs = plt.subplots(1, 2, figsize=(8, 4)) # Create subplots using plt

        axs[0].hist(diameters, bins=15, color='#6f4e37', edgecolor='black')
        axs[0].set_title('Coffee Bean Size (diameter) Distribution')
        axs[0].set_xlabel('Size (mm)')
        axs[0].set_ylabel('Count')

        axs[1].hist(volumes, bins=15, color='#6f4e37', edgecolor='black')
        axs[1].set_title('Coffee Bean Volume Distribution')
        axs[1].set_xlabel('Volume (mm³)')

        plt.tight_layout()
        st.pyplot(fig) # Use st.pyplot to display the matplotlib figure

    return diameters, volumes

# Move all the simulation code inside the Streamlit app section
st.title('HMS lab')

# Add input parameters in the sidebar
with st.sidebar:
    st.header("Simulation Parameters")
    
    # Add file uploader
    uploaded_file = st.file_uploader("Upload experimental data (CSV)", type=['csv'])
    
    # Add weight type selection
    weight_type = st.radio(
        "Weight Type",
        options=['surface', 'volume', 'quantity'],
        index=0,  # Default to 'surface'
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

# st.write("This app simulates a fragmentation process and compares the simulated particle size distribution with measured data.")

# Display Coffee Bean Distribution (optional)
if show_beans:
    st.header('Coffee Bean Initial Distribution')
    bean_display=True

# Move the plot display to main screen
if uploaded_file is not None:
    # Load experimental data from uploaded file
    try:
        exp_data_raw = np.loadtxt(uploaded_file, delimiter=',', encoding='utf-8-sig')
        exp_data = exp_data_raw[(exp_data_raw[:, 0] >= 10) & (exp_data_raw[:, 0] <= 1000)]
        exp_data_x = exp_data[:, 0]/1000
        exp_data_y = exp_data[:, 1]
        exp_data_main = exp_data_raw[(exp_data_raw[:, 0] >= 100) & (exp_data_raw[:, 0] <= 1000)]
        
        # Display the main plot in the main area
        # st.subheader("Data Plot")
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

# Add custom CSS for the mint green button with thicker dark green border
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #98FF98;
        color: black;
        border: 4px solid #228B22;  /* Thicker Forest Green border */
    }
    div.stButton > button:hover {
        background-color: #7FFF7F;
        color: black;
        border: 4px solid #006400;  /* Thicker Darker Green border on hover */
    }
    </style>
""", unsafe_allow_html=True)

# Center the button using columns
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    start_button = st.button(
        'Start Simulation', 
        use_container_width=True,
        type="primary"
    )

# Change if statement to use the new button
if start_button:
    # Add custom CSS for centered header
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
    
    # Run simulation
    _, beans = coffee_beans(150, 4.2, 0.5)
    particles = fragmentation(n_iter, split_ratio_range, beans, epsilon)
    bin_centers, density = calculate_distribution(particles, num_bin=50, weight_type=weight_type)
    sim_data_raw = np.column_stack((10**bin_centers, density))
    sim_data = sim_data_raw[(sim_data_raw[:, 0] >= 0.008) & (sim_data_raw[:, 0] <= 1)]
    sim_data_main = sim_data_raw[(sim_data_raw[:, 0] >= 0.1) & (sim_data_raw[:, 0] <= 1)]

    # Display the main plot

    
    fig_main, ax_main = plt.subplots()
    # ax_main.semilogx(exp_data_x, exp_data_y/np.max(exp_data_y), marker='o', color='red')
    # ax_main.semilogx(sim_data[:,0], sim_data[:,1]/np.max(sim_data[:,1]), marker='.', color='skyblue')

    plt.semilogx(exp_data_x, exp_data_y/np.max(exp_data_main[:,1]), marker='o', color='red')
    plt.semilogx(sim_data[:,0], sim_data[:,1]/np.max(sim_data_main[:,1]), marker='.', color='skyblue')
    ax_main.set_xlabel('Size, Diameter [mm]')
    ax_main.set_ylabel('Normalized value')
    ax_main.legend(['Measured', 'Simulated'])
    ax_main.grid(True, which='both', linestyle='--', linewidth=0.5)
    st.pyplot(fig_main)


