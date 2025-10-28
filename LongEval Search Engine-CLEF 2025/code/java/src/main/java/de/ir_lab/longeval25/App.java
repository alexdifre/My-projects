package de.ir_lab.longeval25;

import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;
import net.sourceforge.argparse4j.inf.Namespace;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.nio.file.Paths;
import java.util.LinkedList;
import java.util.List;
import java.util.zip.GZIPInputStream;
import java.io.*;

public class App {
    public static void main(String[] args) throws IOException {
        Namespace parsedArgs = parseArgs(args);

        List<ObjectNode> queries = parseQueries(parsedArgs.getString("input"));
        List<ObjectNode> documents = parseDocuments(parsedArgs.getString("input"));
        System.out.println("Parsed " + queries.size() + " queries. First one: " + queries.get(0));
        System.out.println("Parsed " + documents.size() + " documents. First one: " + documents.get(0));

        System.out.println("Index Directory: " + parsedArgs.getString("index"));
        System.out.println("ToDo: Store output (e.g., run.txt) in " + parsedArgs.getString("output"));
    }

    private static List<ObjectNode> parseDocuments(String inputDir) throws IOException {
        File file = Paths.get(inputDir, "documents.jsonl.gz").toFile();

        try (final BufferedReader reader = new BufferedReader(new InputStreamReader(new GZIPInputStream(new FileInputStream(file))))) {
            return parseJsonLines(reader);
        }
    }

    private static List<ObjectNode> parseQueries(String inputDir) throws IOException {
        File file = Paths.get(inputDir, "queries.jsonl").toFile();

        try (final BufferedReader reader = new BufferedReader(new FileReader(file))) {
            return parseJsonLines(reader);
        }
    }

    private static List<ObjectNode> parseJsonLines(BufferedReader reader) throws IOException {
        List<ObjectNode> ret = new LinkedList<>();

        String line;
        while ((line = reader.readLine()) != null) {
            ret.add(new ObjectMapper().readValue(line, ObjectNode.class));
        }

        return ret;
    }

    public static Namespace parseArgs(String[] args) {
        ArgumentParser parser = ArgumentParsers.newFor("App").build()
                .defaultHelp(true)
                .description("This is a simple program to retrieve documents from a given index.");

        parser.addArgument("--index")
                .required(true)
                .help("The PyTerrier index.");

        parser.addArgument("--input")
                .required(true)
                .help("The directory containing the input data.");

        parser.addArgument("--output")
                .required(true)
                .help("Place were we store the outputs.");
        try {
            return parser.parseArgs(args);
        } catch (ArgumentParserException e) {
            parser.handleError(e);
            System.exit(1);
            return null;
        }
    }
}
