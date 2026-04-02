import re

with open("Frontend/script.js", "r") as f:
    lines = f.readlines()

new_lines = []
skip = False

replacement_code = """function clearPredictionLabels() {
    if (predictionLabels) {
        predictionLabels.removeAll();
    }
    for (const zoneId in zoneEntities) {
        const entity = zoneEntities[zoneId];
        if (entity && entity.polygon) {
            entity.polygon.material = Cesium.Color.WHITE.withAlpha(0.1);
            entity.polygon.extrudedHeight = 0.0;
            const originalId =
                entity.properties["LocationID"]?.getValue() ||
                entity.properties["location_id"]?.getValue() ||
                "Unknown";
            entity.description = `Zone ID: ${originalId}<br>Select date/time and click 'Predict Demand'.`;
        }
    }
    console.log("🧹 Cleared previous prediction styles.");
}

function displayPredictions(predictions, viewer, selectedDate, selectedTime) {
    if (Object.keys(zoneEntities).length === 0) {
        console.error("❌ Cannot display predictions: Zone entities not loaded.");
        return;
    }

    let totalPredictedDemand = 0;
    let displayedCount = 0;
    let errorCount = 0;
    clearPredictionLabels();
    
    // Find absolute maximum demand for normalisation
    const allDemands = Object.values(predictions).map(Number).filter(d => !isNaN(d) && d > 0);
    const maxExpectedDemand = allDemands.length > 0 ? Math.max(...allDemands, 50) : 150;
    
    for (const zoneIdStr in predictions) {
        try {
            const zoneId = parseInt(zoneIdStr, 10);
            const demand = predictions[zoneIdStr];

            if (isNaN(zoneId)) continue;
            if (demand < 0) {
                errorCount++;
                continue;
            }

            const entity = zoneEntities[zoneId];
            if (entity) {
                totalPredictedDemand += demand;
                displayedCount++;
                
                // HSL Gradient calculation (Yellow to Red)
                const intensity = Math.min(1.0, Math.pow(demand / maxExpectedDemand, 0.7)); // non-linear for better contrast
                
                // Colors: High demand = Red (Hue 0), Low demand = Green (Hue 120) or Yellowish
                const hue = (1.0 - intensity) * 120 / 360; 
                
                entity.polygon.material = Cesium.Color.fromHsl(hue, 0.9, 0.5, 0.85);
                entity.polygon.extrudedHeight = demand * 35.0; // Extrude into 3D!
                entity.polygon.outline = false;
                
                entity.description = `
                    <div style="font-family: 'Nunito Sans', sans-serif;">
                        <h3 style="margin:0 0 5px 0;">Zone ID: ${zoneId}</h3>
                        <p style="margin:0; font-size:16px;"><strong>${demand}</strong> rides predicted</p>
                        <hr style="border:1px solid #444; margin:8px 0;"/>
                        <p style="margin:0; font-size:12px; color:#aaa;">For ${selectedDate} ${selectedTime}</p>
                    </div>
                `;
            }
        } catch (e) {
            console.error(`Error processing prediction for zone ${zoneIdStr}:`, e);
        }
    }

    const statusText = `Prediction for ${selectedDate} ${selectedTime}. <br> Displayed: ${displayedCount}.<br> Total Predicted: ${totalPredictedDemand}.`;
    $("#predictionResult").html(statusText);
    console.log(`✅ ${statusText}`);
    
    viewer.camera.flyTo({
        destination: Cesium.Cartesian3.fromDegrees(-73.985, 40.735, 12000.0),
        orientation: {
            heading: Cesium.Math.toRadians(0.0),
            pitch: Cesium.Math.toRadians(-40.0),
            roll: 0.0
        },
        duration: 3.5,
        easingFunction: Cesium.EasingFunction.QUADRATIC_IN_OUT
    });
}

async function fetchDemandPrediction() {
    console.log("fetchDemandPrediction called");
    const selectedDate = $("#datepicker").val();
    const selectedTime = $("#timepicker").val();

    if (!selectedDate || !selectedTime) {
        $("#predictionResult").text("Please select date and time.");
        return;
    }
    const apiUrl = `${API_BASE_URL}/predict?date=${selectedDate}&time=${selectedTime}`;
    $("#predictionResult").text("Predicting demand...");
    
    let predictButton = document.getElementById("predictButton");
    predictButton.innerText = "⏳ Predicting...";
    predictButton.disabled = true;
    predictButton.style.opacity = "0.7";

    console.log(`Fetching predictions from: ${apiUrl}`);
    clearPredictionLabels();

    try {
        const response = await fetch(apiUrl);
        if (!response.ok) {
            let errorMsg = `API Error (${response.status}): ${response.statusText}`;
            throw new Error(errorMsg);
        }

        const predictions = await response.json();
        console.log("✅ Predictions received");
        displayPredictions(predictions, cesiumViewer, selectedDate, selectedTime);
    } catch (error) {
        console.error("❌ Error fetching or processing predictions:", error);
        $("#predictionResult").text(`Error: ${error.message}`);
        clearPredictionLabels();
    } finally {
        predictButton.innerText = "Predict Demand";
        predictButton.disabled = false;
        predictButton.style.opacity = "1";
    }
}
"""

i = 0
while i < len(lines):
    line = lines[i]
    if line.startswith("function clearPredictionLabels() {"):
        new_lines.append(replacement_code)
        
        # skip lines until the end of fetchDemandPrediction
        while i < len(lines) and not line.startswith("$(document).ready(function () {"):
            i += 1
            if i < len(lines):
                line = lines[i]
        
        new_lines.append(line)
    else:
        new_lines.append(line)
    i += 1

with open("Frontend/script.js", "w") as f:
    f.writelines(new_lines)
