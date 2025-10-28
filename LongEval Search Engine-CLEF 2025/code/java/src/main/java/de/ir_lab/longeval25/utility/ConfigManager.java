package de.ir_lab.longeval25.utility;


import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Map;
import java.util.Properties;

import org.yaml.snakeyaml.Yaml;

public class ConfigManager {
    private static ConfigManager instance;
    private final Map<String, Object> config;

    private ConfigManager() {
        Yaml yaml = new Yaml();
        try (InputStream inputStream = new FileInputStream( "code/java/src/main/resources/params.yml")) {
            config = yaml.load(inputStream);
        } catch (IOException e) {
            throw new IllegalStateException("Errore nella lettura del file YAML", e);
        }
    }

    public static synchronized ConfigManager getInstance() {
        if (instance == null) {
            instance = new ConfigManager();
        }
        return instance;
    }

    // Metodo per ottenere il valore come stringa
    public String getString(String key) {
        Object value = config.get(key);
        return value != null ? value.toString() : null;
    }

    // Metodo per ottenere il valore come numero intero
    public Integer getInt(String key) {
        Object value = config.get(key);
        return (value instanceof Number number) ? number.intValue() : null;
    }

    // Metodo per ottenere il valore come boolean
    public Boolean getBool(String key) {
        Object value = config.get(key);
        return (value instanceof Boolean bool) ? bool : null;
    }
    
    // Metodo per ottenere il valore come double
    public Double getDouble(String key) {
        Object value = config.get(key);
        return (value instanceof Number number) ? number.doubleValue() : null;
    }

    // Metodo per recuperare le api key per i servizi
    public String getOpenApiKey() throws IOException {
        Properties property = new Properties();
        try (FileInputStream input = new FileInputStream("code/java/src/main/resources/config.properties")) {
            property.load(input);
        }
        return property.getProperty("api.key");
    }
    
}
