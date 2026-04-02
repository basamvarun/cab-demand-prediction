import re

with open("Frontend/script.js", "r") as f:
    content = f.read()

# 1. Update Cesium Viewer to hide UI clutter
content = content.replace(
    '''                cesiumViewer = new Cesium.Viewer("cesiumContainer", {
                    terrain: Cesium.Terrain.fromWorldTerrain(),
                    geocoder: false,
                });''',
    '''                document.getElementById("parent").style.display = "flex"; // Ensure container has size
                cesiumViewer = new Cesium.Viewer("cesiumContainer", {
                    terrain: Cesium.Terrain.fromWorldTerrain(),
                    geocoder: false,
                    animation: false,
                    timeline: false,
                    baseLayerPicker: false,
                    selectionIndicator: false,
                    navigationHelpButton: false
                });'''
)

# 2. Fix init loading logic
intro_logic_old = """$(document).ready(function () {
    console.log("Document ready. Initializing...");
    setTimeout(function () {
        $("#intro-container").css("opacity", "0");
        setTimeout(function () {
            $("#intro-container").css("display", "none");
            $("#parent").css("display", "flex");
            console.log("Intro finished. Setting up main UI and Cesium...");"""

intro_logic_new = """$(document).ready(function () {
    console.log("Document ready. Initializing...");
    // Intro logic will hide once data is loaded instead of randomly waiting
    console.log("Setting up main UI and Cesium...");"""

content = content.replace(intro_logic_old, intro_logic_new)

# 3. Modify end of Document Ready logic to resolve Intro promise
old_end = """            let removeButton = document.getElementById("removePredictButton");
            if (removeButton) {
                removeButton.addEventListener("click", function () {
                    console.log("🧹 Remove Predictions button clicked!");
                    clearPredictionLabels();
                    $("#predictionResult").text("Predictions cleared.");
                });
                console.log("✅ Remove Predictions button listener attached.");
            } else {
                console.error("❌ Remove Predict Button not found in the DOM.");
            }
        }, 500);
    }, 3000);
});"""

new_end = """            let removeButton = document.getElementById("removePredictButton");
            if (removeButton) {
                removeButton.addEventListener("click", function () {
                    console.log("🧹 Remove Predictions button clicked!");
                    clearPredictionLabels();
                    $("#predictionResult").text("Predictions cleared.");
                });
                console.log("✅ Remove Predictions button listener attached.");
            } else {
                console.error("❌ Remove Predict Button not found in the DOM.");
            }

            // --- Wait for loadZones to finish, then dismiss intro screen ---
            const checkReady = setInterval(() => {
                if (zoneDataSource !== null) { // we know it's loaded
                    clearInterval(checkReady);
                    $("#intro-container").css("opacity", "0");
                    setTimeout(function () {
                        $("#intro-container").css("display", "none");
                    }, 500);
                }
            }, 300);

});"""
content = content.replace(old_end, new_end)


with open("Frontend/script.js", "w") as f:
    f.write(content)
print("Updated basic init logic")
